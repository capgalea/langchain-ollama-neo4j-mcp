import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from main_multi import get_all_tools, MCP_SERVER_CONFIGS

async def test_tool_names():
    print("Testing tool name sanitization...")
    print()
    
    tools = await get_all_tools(MCP_SERVER_CONFIGS)
    
    print("\nTool names validation:")
    import re
    pattern = r'^[a-zA-Z0-9_-]{1,128}$'
    
    for tool in tools:
        matches = re.match(pattern, tool.name)
        status = "✅" if matches else "❌"
        print(f"{status} {tool.name} - Valid: {bool(matches)}")

if __name__ == "__main__":
    asyncio.run(test_tool_names())
