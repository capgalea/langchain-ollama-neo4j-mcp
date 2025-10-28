import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from main_simple import run_agent

async def test_mcp():
    print("Testing MCP server connection to Neo4j...")
    print(f"Neo4j URI: {os.getenv('NEO4J_URI')}")
    print()
    
    # Test query
    request = "How many nodes are in the database?"
    model = "claude-3-5-sonnet-20241022"
    
    print(f"Query: {request}")
    print(f"Model: {model}")
    print("Processing...")
    print()
    
    try:
        result = await run_agent(request, model)
        print("SUCCESS: MCP Query successful!")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nRaw response length: {len(str(result['raw']))}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp())
