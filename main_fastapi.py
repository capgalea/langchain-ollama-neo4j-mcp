from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import os

# Load environment variables first
load_dotenv()

# Import MultiToolAgent after dotenv is loaded and only when needed
# This helps avoid multiprocessing issues on Windows
def get_multi_tool_agent():
    from main_multi import MultiToolAgent, MCP_SERVER_CONFIGS
    return MultiToolAgent, MCP_SERVER_CONFIGS

# Get the fastapi logger
logger = logging.getLogger("fastapi")

# Get configuration from environment variables with defaults
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Create FastAPI app with configuration
app = FastAPI(
    title="LangChain + Ollama + Neo4j MCP API",
    description="API for interacting with LangChain and Neo4j through MCP with Ollama",
    version=os.getenv("API_VERSION", "1.0.0"),
    docs_url=os.getenv("DOCS_URL", "/docs"),
    redoc_url=os.getenv("REDOC_URL", "/redoc"),
    openapi_url=os.getenv("OPENAPI_URL", "/openapi.json"),
    # Force the server to use our host and port
    servers=[{"url": f"http://{FASTAPI_HOST}:{FASTAPI_PORT}", "description": "Local Development"}]
)

# Enable CORS with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=os.getenv("CORS_METHODS", "*").split(","),
    allow_headers=os.getenv("CORS_HEADERS", "*").split(","),
)



@app.get("/query")
async def query_agent(
    command: str = Query(..., 
        description="Simple instruction for the graph database agent", 
        example="Find all grants associated with the researcher 'raymond norton'."
    ), model: str = Query(..., 
        description="The name of the Ollama model to use. NOTE: Model must be available on the ollama server.", 
        example="llama3.1"
    )):
    """
    Execute a command through the LangChain agent with Neo4j MCP integration.
    
    Args:
        command (str): The command to be executed by the agent
        
    Returns:
        dict: The response from the agent
    """
    if not command:
        raise HTTPException(status_code=400, detail="Command parameter is required")
    
    try:
        # Get or create agent from cache
        agent = get_agent(model)
        result = await agent.run_request(command, with_logging=False)  # Enable logging for API requests
        
        # Get the raw response
        raw_response = result.get("raw", "")
        
        # Try to extract Cypher query from the raw agent response
        cypher_query = ""
        try:
            print(f"\n=== CYPHER EXTRACTION DEBUG ===")
            print(f"Model: {model}")
            print(f"Raw response type: {type(raw_response)}")
            print(f"Has 'get' method: {hasattr(raw_response, 'get')}")
            
            # Strategy 1: Check if raw_response has messages attribute (LangGraph response)
            if hasattr(raw_response, 'get'):
                messages = raw_response.get('messages', [])
                print(f"Found {len(messages)} messages in raw_response")
                
                # Look through messages for AIMessage with tool_calls
                for idx, msg in enumerate(messages):
                    print(f"  Message {idx}: type={type(msg)}, has_tool_calls={hasattr(msg, 'tool_calls')}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"    Tool calls count: {len(msg.tool_calls)}")
                        for tc_idx, tool_call in enumerate(msg.tool_calls):
                            tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                            print(f"      Tool call {tc_idx}: type={type(tool_call)}, name={tool_name}")
                            
                            # Handle dict format
                            if isinstance(tool_call, dict):
                                if tool_call.get('name') == 'read_neo4j_cypher':
                                    cypher_query = tool_call.get('args', {}).get('query', '')
                                    print(f"✓ Extracted Cypher from dict tool_call: {len(cypher_query)} chars")
                                    break
                            # Handle object format
                            elif hasattr(tool_call, 'name') and hasattr(tool_call, 'args'):
                                if tool_call.name == 'read_neo4j_cypher':
                                    # args might be a dict or an object
                                    if isinstance(tool_call.args, dict):
                                        cypher_query = tool_call.args.get('query', '')
                                    elif hasattr(tool_call.args, 'query'):
                                        cypher_query = tool_call.args.query
                                    print(f"✓ Extracted Cypher from object tool_call: {len(cypher_query)} chars")
                                    break
                    if cypher_query:
                        break
            
            # Strategy 2: Search in string representation if not found
            if not cypher_query and raw_response:
                print(f"\n  Trying regex extraction from string representation...")
                import re
                raw_str = str(raw_response)
                print(f"  String length: {len(raw_str)} chars")
                print(f"  First 500 chars: {raw_str[:500]}")
                
                # Look for read_neo4j_cypher with query parameter
                patterns = [
                    r"'name':\s*'read_neo4j_cypher'.*?'query':\s*'((?:[^'\\]|\\.)*)'\s*[,}]",
                    r'"name":\s*"read_neo4j_cypher".*?"query":\s*"((?:[^"\\]|\\.)*)"',
                ]
                for pattern_idx, pattern in enumerate(patterns):
                    match = re.search(pattern, raw_str, re.DOTALL)
                    if match:
                        cypher_query = match.group(1)
                        # Unescape
                        cypher_query = cypher_query.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
                        print(f"✓ Extracted Cypher via regex pattern {pattern_idx+1}: {len(cypher_query)} chars")
                        break
            
            print(f"=== Final extraction result: {'SUCCESS' if cypher_query else 'FAILED'} ===")
            if cypher_query:
                print(f"Query preview (first 200 chars): {cypher_query[:200]}")
            print(f"==============================\n")
                        
        except Exception as extract_error:
            print(f"Error extracting Cypher query: {extract_error}")
            import traceback
            traceback.print_exc()
        
        # Ensure all values are JSON serializable
        response = {
            "status": "success", 
            "result": str(result.get("answer", "")),  # Convert to string to ensure serialization
            "raw": str(raw_response),        # Convert raw to string
            "cypher_query": cypher_query,  # Send extracted query separately
            "seconds_to_complete": float(result.get("seconds_to_complete", 0.0))     # Explicitly convert to float
        }
        print(f"API Response: {response}")
        return response
    except Exception as e:
        print(f"Error in query_agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema():
    """
    Get the Neo4j database schema.
    
    Returns:
        list: The schema information from the database
    """
    try:
        # Use a default agent to fetch schema
        agent = get_agent("qwen2.5:1.5b")  # Use lightweight model for schema fetch
        if not agent.agent:
            await agent.initialize()
        
        # Return cached schema if available
        if hasattr(agent, 'schema_data') and agent.schema_data:
            return agent.schema_data
        
        return []
    except Exception as e:
        print(f"Error in get_schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache for agents by model name
_agent_cache = {}

def get_agent(model: str):
    """
    Get a cached agent instance or create a new one if it doesn't exist.
    
    Args:
        model: The model name to get or create an agent for
        
    Returns:
        MultiToolAgent: A cached or new agent instance
    """
    if model not in _agent_cache:
        MultiToolAgent, MCP_SERVER_CONFIGS = get_multi_tool_agent()
        _agent_cache[model] = MultiToolAgent(model, MCP_SERVER_CONFIGS)
    return _agent_cache[model]

# This allows the file to be imported without starting the server
if __name__ == "__main__":
    import uvicorn
    
    # On Windows with Python 3.13, run without reload to avoid multiprocessing issues
    print("🚀 Starting FastAPI server on http://127.0.0.1:8002")
    print("📖 API docs: http://127.0.0.1:8002/docs")
    
    # Run the server with our configuration
    uvicorn.run(
        app,  # Pass app object directly instead of string to avoid reload issues
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        log_level="info"
    )
