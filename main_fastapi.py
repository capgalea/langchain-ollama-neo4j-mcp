from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from main_multi import MultiToolAgent, MCP_SERVER_CONFIGS
from dotenv import load_dotenv
import logging
import os


# Get the fastapi logger
logger = logging.getLogger("fastapi")

# Load environment variables
load_dotenv()

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
        example="Create a new node with the label 'Person' and the property 'name' set to 'John Doe'."
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
        
        # Ensure all values are JSON serializable
        response = {
            "status": "success", 
            "result": str(result.get("answer", "")),  # Convert to string to ensure serialization
            "raw": str(result.get("raw", "")),        # Convert raw to string
            "seconds_to_complete": float(result.get("seconds_to_complete", 0.0))     # Explicitly convert to float
        }
        print(f"API Response: {response}")
        return response
    except Exception as e:
        print(f"Error in query_agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache for agents by model name
_agent_cache = {}

def get_agent(model: str) -> MultiToolAgent:
    """
    Get a cached agent instance or create a new one if it doesn't exist.
    
    Args:
        model: The model name to get or create an agent for
        
    Returns:
        MultiToolAgent: A cached or new agent instance
    """
    if model not in _agent_cache:
        _agent_cache[model] = MultiToolAgent(model, MCP_SERVER_CONFIGS)
    return _agent_cache[model]

# This allows the file to be imported without starting the server
if __name__ == "__main__":
    import uvicorn
    
    # Run the server with our configuration
    uvicorn.run(
        "main_fastapi:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=os.getenv("FASTAPI_RELOAD", "true").lower() == "true"
    )
