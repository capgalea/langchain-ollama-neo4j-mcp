from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks import get_openai_callback
from langchain_core.tracers import ConsoleCallbackHandler
import asyncio
import time
import os

from dotenv import load_dotenv
load_dotenv()

# Extending from earlier simple example
from main_simple import get_model, interpret_agent_response

# Define multiple MCP servers
MCP_SERVER_CONFIGS = {
        "neo4j-cypher": {
            "command": "uvx",
            "args": ["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
            "transport": "stdio",
            "env": os.environ
        },
        "neo4j-data-modeling": {
            "command": "uvx",
            "args": ["mcp-neo4j-data-modeling@0.1.1", "--transport", "stdio"],
            "transport": "stdio",
            "env": os.environ
        },
        "memory": {
            "command": "uvx",
            "args": [ "mcp-neo4j-memory@0.1.5" ],
            "transport": "stdio",
            "env": os.environ
        },
        # Requires a paid Aura account
        # "neo4j-aura": {
        #     "command": "uvx",
        #     "args": [ "mcp-neo4j-aura-manager@0.2.2" ],
        #     "transport": "stdio",
        #     "env": os.environ
        # }
    }

# LESS ELEGANT but workable way to define multiple MCP servers - stdio only
async def get_tools_from_server(server_name: str, server_cfg: dict):
    """Setup stdio tools for a single server configuration"""
    params = StdioServerParameters(
        command=server_cfg["command"],
        args=server_cfg["args"],
        transport=server_cfg["transport"],
        env=server_cfg["env"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools
        
async def get_all_tools(configs: dict):
    """Get all tools from all servers"""
    # Get all tools from all servers in parallel
    all_tools_lists = await asyncio.gather(*[
        get_tools_from_server(name, cfg) for name, cfg in configs.items()
    ])
    # Flatten the list of lists
    all_tools = [tool for tools in all_tools_lists for tool in tools]

    print("\nAvailable tools:")
    for tool in all_tools:
        print(f"- {tool.name}: {tool.description}")

    return all_tools

# MORE ELEGANT way to define multiple MCP servers - supports stdio, sse, and streamable-http
async def get_multi_tools(configs: list[dict]):
    client = MultiServerMCPClient(configs)
    return await client.get_tools()


class MultiToolAgent:
    def __init__(self, model: str, configs: list[dict]):
        self.model = model
        self.configs = configs
        self.agent = None
        self.tools = None

    async def initialize(self):
        """Initialize the agent with tools from all configured servers"""
        # LESS ELEGANT approach
        # self.tools = await get_all_tools(self.configs)

        # MORE ELEGANT approach
        self.tools = await get_multi_tools(self.configs)
        self.agent = create_react_agent(get_model(self.model), self.tools)
        return self

    async def run_request(self, request: str, with_logging: bool = False) -> dict:
        """Internal method to process a request with optional logging"""
        if not self.agent:
            await self.initialize()

        start_time = time.time()
        
        if with_logging:
            print(f"\n{'='*50}\nProcessing request: {request}\n{'='*50}")
            callbacks = [ConsoleCallbackHandler()]
            
            if 'gpt' in self.model.lower():
                with get_openai_callback() as cb:
                    agent_response = await self.agent.ainvoke(
                        {"messages": request},
                        {"callbacks": callbacks}
                    )
                    print(f"\nToken usage: {cb}")
            else:
                agent_response = await self.agent.ainvoke(
                    {"messages": request},
                    {"callbacks": callbacks}
                )
            
            print(f"\n{'='*50}\nRaw response:\n{agent_response}\n{'='*50}")
            interpreted = await interpret_agent_response(agent_response, request, self.model)
            print(f"\n{'='*50}\nFinal answer:\n{interpreted}\n{'='*50}")
        else:
            agent_response = await self.agent.ainvoke({"messages": request})
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