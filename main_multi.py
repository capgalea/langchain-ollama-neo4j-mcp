from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks import get_openai_callback
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
import asyncio
import time
import os

from dotenv import load_dotenv
load_dotenv()

# Extending from earlier simple example
from main_simple import get_model, interpret_agent_response

# Define multiple MCP servers
MCP_SERVER_CONFIGS = {
    # "neo4j-cypher": {
    #     "command": "uvx",
    #     "args": ["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
    #     "transport": "stdio",
    #     "env": os.environ
    # }
    # Commented out additional servers to test one at a time
    # "neo4j-data-modeling": {
    #     "command": "uvx",
    #     "args": ["mcp-neo4j-data-modeling@0.1.1", "--transport", "stdio"],
    #     "transport": "stdio",
    #     "env": os.environ
    # },
    # "memory": {
    #     "command": "uvx",
    #     "args": [ "mcp-neo4j-memory@0.1.5" ],
    #     "transport": "stdio",
    #     "env": os.environ
    # },
    # Requires a paid Aura account
    "neo4j-aura": {
        "command": "uvx",
        "args": ["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
        "transport": "stdio",
        "env": os.environ
    }
}

# LESS ELEGANT but workable way to define multiple MCP servers - stdio only
async def get_tools_from_server(server_name: str, server_cfg: dict):
    """Setup stdio tools for a single server configuration"""
    params = StdioServerParameters(
        command=server_cfg["command"],
        args=server_cfg["args"],
        env=server_cfg["env"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools
        
def sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match Claude's pattern: ^[a-zA-Z0-9_-]{1,128}$"""
    import re
    # Replace any invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Ensure it's within 128 characters
    return sanitized[:128]

async def get_all_tools(configs: dict):
    """Get all tools from all servers"""
    # Get all tools from all servers in parallel
    all_tools_lists = await asyncio.gather(*[
        get_tools_from_server(name, cfg) for name, cfg in configs.items()
    ])
    # Flatten the list of lists
    all_tools = [tool for tools in all_tools_lists for tool in tools]

    # Sanitize tool names for Claude compatibility
    for tool in all_tools:
        original_name = tool.name
        tool.name = sanitize_tool_name(tool.name)
        if original_name != tool.name:
            print(f"Sanitized tool name: {original_name} -> {tool.name}")

    print("\nAvailable tools:")
    for tool in all_tools:
        print(f"- {tool.name}: {tool.description}")

    return all_tools

# MORE ELEGANT way to define multiple MCP servers - supports stdio, sse, and streamable-http
async def get_multi_tools(configs: dict):
    client = MultiServerMCPClient(configs)
    return await client.get_tools()


class MultiToolAgent:
    def __init__(self, model: str, configs: dict):
        self.model = model
        self.configs = configs
        self.agent = None
        self.tools = None
        self._sessions = []  # Keep track of active sessions
        self._clients = []   # Keep track of active clients

    async def initialize(self):
        """Initialize the agent with tools from all configured servers"""
        try:
            # Create new sessions and keep them alive
            all_tools = []
            
            for name, cfg in self.configs.items():
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg["args"],
                    env=cfg["env"]
                )
                
                # Create and store client and session
                client = stdio_client(params)
                read, write = await client.__aenter__()
                self._clients.append((client, read, write))
                
                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()
                self._sessions.append(session)
                
                # Load tools from this session
                tools = await load_mcp_tools(session)
                all_tools.extend(tools)
            
            # Sanitize tool names for Claude compatibility
            for tool in all_tools:
                original_name = tool.name
                tool.name = sanitize_tool_name(tool.name)
                if original_name != tool.name:
                    print(f"Sanitized tool name: {original_name} -> {tool.name}")
            
            print("\nAvailable tools:")
            for tool in all_tools:
                print(f"- {tool.name}: {tool.description}")
            
            # Fetch schema to provide context to the agent
            schema_context = ""
            schema_data = None
            try:
                schema_tool = None
                for tool in all_tools:
                    if 'schema' in tool.name.lower():
                        schema_tool = tool
                        break
                
                if schema_tool:
                    print("\nFetching database schema...")
                    schema_data = await schema_tool.ainvoke({})
                    # Format schema for LLM context
                    schema_context = self._format_schema_context(schema_data)
                    print("✓ Schema loaded")
            except Exception as e:
                print(f"Warning: Could not fetch schema: {e}")
            
            self.tools = all_tools
            self.schema_data = schema_data  # Store raw schema data
            self.schema_context = schema_context
            self.agent = create_react_agent(get_model(self.model), self.tools)
            return self
        except Exception as e:
            print(f"Error initializing agent: {e}")
            import traceback
            traceback.print_exc()
            # Clean up on error
            await self.cleanup()
            raise

    async def cleanup(self):
        """Clean up all MCP sessions and clients"""
        for session in self._sessions:
            try:
                await session.__aexit__(None, None, None)
            except:
                pass
        
        for client, read, write in self._clients:
            try:
                await client.__aexit__(None, None, None)
            except:
                pass
        
        self._sessions = []
        self._clients = []

    def _format_schema_context(self, schema_data):
        """Format schema data for LLM context - ultra-concise version"""
        if not schema_data:
            return ""
        
        # Create minimal schema summary
        context = "\n\nNeo4j Schema Summary:\n"
        node_types = []
        
        for node_info in schema_data:
            label = node_info.get('label', 'Unknown')
            attrs = node_info.get('attributes', {})
            rels = node_info.get('relationships', {})
            
            # Only show unique/indexed properties (most important ones)
            key_props = [k for k, v in attrs.items() if 'unique' in str(v).lower() or 'indexed' in str(v).lower()]
            if not key_props and attrs:
                # If no unique props, just show first 2 properties
                key_props = list(attrs.keys())[:2]
            
            # Format: NodeType(key_property) -[REL]-> TargetType
            rel_summary = ", ".join(f"-[{k}]->{v}" for k, v in list(rels.items())[:3]) if rels else ""
            prop_summary = f"({', '.join(key_props[:2])})" if key_props else ""
            
            node_types.append(f"{label}{prop_summary} {rel_summary}".strip())
        
        context += "; ".join(node_types)
        context += "\nUse get_neo4j_schema tool for full details if needed."
        context += "\n\nIMPORTANT: When creating Cypher queries, ALWAYS use toLower() for text comparisons to ensure case-insensitive matching."
        context += "\nExample: WHERE toLower(r.CI_Name) CONTAINS toLower('tony velkov')"
        
        return context

    async def run_request(self, request: str, with_logging: bool = False) -> dict:
        """Internal method to process a request with optional logging"""
        if not self.agent:
            await self.initialize()
        
        if not self.agent:
            raise RuntimeError("Agent initialization failed - agent is still None")

        # Only inject schema context for database-related queries (to avoid token limits)
        # Check if query mentions database-related terms
        db_keywords = ['grant', 'researcher', 'organization', 'funding', 'field', 'how many', 'count', 'find', 'show', 'list', 'get', 'database']
        should_include_schema = any(keyword in request.lower() for keyword in db_keywords)
        
        full_request = request
        if should_include_schema:
            # Add strong instruction for case-insensitive matching and table formatting
            cypher_instruction = "\n\n" + "="*80 + "\n"
            cypher_instruction += "CRITICAL CYPHER QUERY INSTRUCTIONS (APPLY TO ALL QUERIES):\n"
            cypher_instruction += "="*80 + "\n\n"
            
            cypher_instruction += "0. QUERY COMPLETENESS (ABSOLUTE PRIORITY - READ CAREFULLY):\n"
            cypher_instruction += "   ⚠️ CRITICAL: You MUST answer the COMPLETE question - do NOT return partial data\n"
            cypher_instruction += "   ⚠️ If asked for 'grant details', you MUST return grant properties, NOT just researcher info\n"
            cypher_instruction += "   ⚠️ If asked for 'grants for researcher X', you MUST:\n"
            cypher_instruction += "      1. MATCH the researcher node\n"
            cypher_instruction += "      2. MATCH their relationship to grants: (r)-[rel:CHIEF_INVESTIGATOR_ON|COLLABORATOR_ON]->(g:Grant)\n"
            cypher_instruction += "      3. RETURN grant details: g.Application_ID, g.Grant_Title, g.Total_Amount, etc.\n"
            cypher_instruction += "   \n"
            cypher_instruction += "   WRONG EXAMPLES (THESE ARE INCOMPLETE):\n"
            cypher_instruction += "   × MATCH (r:Researcher) WHERE ... RETURN r.CI_Name  -- Missing grants!\n"
            cypher_instruction += "   × MATCH (r:Researcher) RETURN r  -- Not returning grant details!\n"
            cypher_instruction += "   \n"
            cypher_instruction += "   CORRECT EXAMPLE:\n"
            cypher_instruction += "   ✓ MATCH (r:Researcher)-[rel:CHIEF_INVESTIGATOR_ON|COLLABORATOR_ON]->(g:Grant)\n"
            cypher_instruction += "     WHERE toLower(r.CI_Name) CONTAINS toLower('name')\n"
            cypher_instruction += "     RETURN DISTINCT r.CI_Name, type(rel) as Role, g.Application_ID, g.Grant_Title, g.Total_Amount\n\n"
            
            cypher_instruction += "1. CASE-INSENSITIVE MATCHING:\n"
            cypher_instruction += "   - ALWAYS use toLower() for ALL text comparisons\n"
            cypher_instruction += "   - Example: WHERE toLower(property) CONTAINS toLower('search term')\n"
            cypher_instruction += "   - Applies to: CONTAINS, =, STARTS WITH, ENDS WITH, etc.\n\n"
            
            cypher_instruction += "2. RELATIONSHIP TYPE FUNCTION:\n"
            cypher_instruction += "   - type() ONLY works on relationships, NOT nodes\n"
            cypher_instruction += "   - CORRECT: MATCH (a)-[rel:REL_TYPE]->(b) RETURN type(rel)\n"
            cypher_instruction += "   - WRONG: MATCH (a:Node) RETURN type(a)\n\n"
            
            cypher_instruction += "3. AVOID DUPLICATE RESULTS:\n"
            cypher_instruction += "   - When matching multiple relationship types [:TYPE1|TYPE2], ALWAYS use DISTINCT\n"
            cypher_instruction += "   - REQUIRED: MATCH (r)-[:TYPE1|TYPE2]->(g) RETURN DISTINCT ...\n"
            cypher_instruction += "   - Or use single relationship type if duplicates not wanted\n\n"
            
            cypher_instruction += "4. TABLE FORMAT (CRITICAL - ABSOLUTE REQUIREMENT):\n"
            cypher_instruction += "   - FORBIDDEN: NEVER EVER use COLLECT() in RETURN clause\n"
            cypher_instruction += "   - FORBIDDEN: NEVER create nested objects or maps in RETURN\n"
            cypher_instruction += "   - REQUIRED: Return ONE ROW per grant/entity with ONLY scalar values\n"
            cypher_instruction += "   - REQUIRED: Use simple properties directly: g.Grant_Title, g.Total_Amount, etc.\n"
            cypher_instruction += "   - If showing both CI and Collaborator grants, use separate MATCH or UNION\n"
            cypher_instruction += "   - CORRECT EXAMPLE:\n"
            cypher_instruction += "     MATCH (r:Researcher)-[rel:CHIEF_INVESTIGATOR_ON|COLLABORATOR_ON]->(g:Grant)\n"
            cypher_instruction += "     WHERE toLower(r.CI_Name) CONTAINS toLower('name')\n"
            cypher_instruction += "     RETURN DISTINCT r.CI_Name, type(rel) as Role, g.Application_ID, g.Grant_Title, g.Total_Amount\n"
            cypher_instruction += "   - WRONG EXAMPLES (DO NOT DO THIS):\n"
            cypher_instruction += "     × COLLECT({title: g.Grant_Title}) -- Creates nested structure\n"
            cypher_instruction += "     × COLLECT(DISTINCT g.Grant_Title) -- Creates list\n"
            cypher_instruction += "     × {role: 'CI', title: g.Grant_Title} -- Creates map object\n\n"
            
            cypher_instruction += "5. PROPERTY EXISTENCE (CRITICAL - SYNTAX CHANGE):\n"
            cypher_instruction += "   - NEVER use exists() function - it's deprecated and causes errors\n"
            cypher_instruction += "   - ALWAYS use 'IS NOT NULL' or 'IS NULL' instead\n"
            cypher_instruction += "   - CORRECT: WHERE r.CI_Name IS NOT NULL\n"
            cypher_instruction += "   - WRONG: WHERE exists(r.CI_Name) -- THIS WILL FAIL\n"
            cypher_instruction += "   - Note: In most cases, checking existence is unnecessary - just use the property directly\n\n"
            
            cypher_instruction += "6. SINGLE STATEMENT ONLY:\n"
            cypher_instruction += "   - Return ONLY ONE Cypher statement - no semicolons, no multiple queries\n"
            cypher_instruction += "   - CORRECT: MATCH (r:Researcher) RETURN r.CI_Name\n"
            cypher_instruction += "   - WRONG: MATCH (r:Researcher); RETURN r.CI_Name; -- Multiple statements not allowed\n\n"
            
            cypher_instruction += "="*80 + "\n\n"
            
            if hasattr(self, 'schema_context') and self.schema_context:
                full_request = self.schema_context + cypher_instruction + "\n\nUSER REQUEST: " + request
            else:
                full_request = cypher_instruction + "\n\nUSER REQUEST: " + request

        start_time = time.time()
        if with_logging:
            print(f"\n{'='*50}\nProcessing request: {request}\n{'='*50}")
            callbacks: list[BaseCallbackHandler] = [ConsoleCallbackHandler()]
            callbacks = [ConsoleCallbackHandler()]
            if 'gpt' in self.model.lower():
                with get_openai_callback() as cb:
                    agent_response = await self.agent.ainvoke(
                        {"messages": full_request},
                        config=RunnableConfig(callbacks=callbacks)
                    )
                    print(f"\nToken usage: {cb}")
            else:
                agent_response = await self.agent.ainvoke(
                    {"messages": full_request},
                    config=RunnableConfig(callbacks=callbacks)
                )

            
            print(f"\n{'='*50}\nRaw response:\n{agent_response}\n{'='*50}")
            interpreted = await interpret_agent_response(agent_response, request, self.model)
            print(f"\n{'='*50}\nFinal answer:\n{interpreted}\n{'='*50}")
        else:
            agent_response = await self.agent.ainvoke({"messages": full_request})
            interpreted = await interpret_agent_response(agent_response, request, self.model)
        
        return {
            "raw": agent_response,
            "answer": interpreted,
            "seconds_to_complete": round(time.time() - start_time, 2)  # Rounded to 2 decimal places
        }


# Run the async function
if __name__ == "__main__":

    # Edit the model name here - run `ollama list` to see available models
    model = "llama3.1"
    
    # Write request
    # request = "Create a new node with the label 'Person' and the property 'name' set to 'John Doe'."
    
    # Read request
    request = "How many nodes are in the graph?"

    async def main():
        print("Initializing agent...")
        agent = await MultiToolAgent(model, MCP_SERVER_CONFIGS).initialize()
        print("Processing request...")
        result = await agent.run_request(request)
        print(result)

    asyncio.run(main())