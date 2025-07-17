from main_multi import MultiToolAgent, MCP_SERVER_CONFIGS
import asyncio

async def interactive_agent(agent: any):
    # Setup interactive loop
    print("\nType your request (or 'exit' to quit):")
    while True:
        user_input = input("👶 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting interactive session.")
            break
        try:
            # Use the version with detailed logging
            agent_response = await agent.run_request(user_input)
            # Just print the answer part in the interactive session
            print(f"\n🤖 Agent: {agent_response.get('answer', 'No answer provided')}\n")
        except Exception as e:
            print(f"\nError processing request: {str(e)}\n")

# Run the async function
if __name__ == "__main__":

    # Edit the model name here - run `ollama list` to see available models
    model = "llama3.1"

    async def main():
        agent = await MultiToolAgent(model, MCP_SERVER_CONFIGS).initialize()
        await interactive_agent(agent)

    asyncio.run(main())