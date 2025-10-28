import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from main_multi import MultiToolAgent, MCP_SERVER_CONFIGS

async def test_researcher_query():
    """Test a specific query about researchers"""
    print("Testing researcher count query...")
    print()
    
    # Test with different models
    models = ["qwen2.5:1.5b", "gpt-4o-mini"]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing with model: {model_name}")
        print('='*60)
        
        try:
            agent = MultiToolAgent(model_name, MCP_SERVER_CONFIGS)
            await agent.initialize()
            
            request = "How many researchers are in the database?"
            print(f"Query: {request}")
            print()
            
            result = await agent.run_request(request, with_logging=False)
            
            print(f"Answer: {result.get('answer', 'No answer')}")
            print(f"\nTime taken: {result.get('seconds_to_complete', 0):.2f} seconds")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    asyncio.run(test_researcher_query())
