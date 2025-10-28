import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def test_mcp_tools():
    print("Testing MCP server and tools...")
    print(f"Neo4j URI: {os.getenv('NEO4J_URI')}")
    print()
    
    # MCP server configuration
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
        env=dict(os.environ)
    )
    
    print("Connecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("✅ MCP client connected")
            
            # Initialize session
            await session.initialize()
            print("✅ Session initialized")
            
            # Load tools
            tools = await load_mcp_tools(session)
            print(f"✅ Loaded {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test a tool directly
            if tools:
                print(f"\n🔧 Testing tool: {tools[0].name}")
                try:
                    result = await tools[0].ainvoke({"query": "MATCH (n) RETURN count(n) as count"})
                    print(f"✅ Tool execution successful!")
                    print(f"Result: {result}")
                except Exception as e:
                    print(f"❌ Tool execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
