import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def test_query_execution():
    print("Testing MCP tool query execution...")
    print(f"Neo4j URI: {os.getenv('NEO4J_URI')}")
    print()
    
    # MCP server configuration
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
        env=dict(os.environ)
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            # Find the read_neo4j_cypher tool
            read_tool = None
            for tool in tools:
                if 'read' in tool.name.lower() and 'cypher' in tool.name.lower():
                    read_tool = tool
                    break
            
            if not read_tool:
                print("ERROR: Could not find read_neo4j_cypher tool")
                return
            
            print(f"Found tool: {read_tool.name}")
            print()
            
            # Test query: Count grants
            query = "MATCH (n:Grant) RETURN count(n) as grant_count"
            print(f"Executing query: {query}")
            print()
            
            try:
                result = await read_tool.ainvoke({"query": query})
                print("SUCCESS: Query executed!")
                print(f"Result: {result}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_query_execution())
