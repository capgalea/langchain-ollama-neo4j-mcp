from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

# Create single server parameter for stdio connection
server_params = StdioServerParameters(
    command="uvx",
    args=["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
    transport="stdio",
    env=os.environ
)

def get_model(model_name):
    return ChatOllama(
        model=model_name,
        temperature=0.0,
        streaming=False  # Disable streaming for better compatibility
    )

def extract_content(val):
    # Helper to get .content if present, else str
    return val.content if hasattr(val, "content") else str(val)

async def interpret_agent_response(agent_response, request, model_name="llama3.1"):

    # Use LLM to interpret
    llm = get_model(model_name)
    prompt = (
        "You are an expert assistant. Given the following user request and the raw agent/tool response, "
        "return the most appropriate response to answer the user request."
        f"User request: {request}\n"
        f"Agent/tool response: {agent_response}\n"
        "Answer:"
    )
    if hasattr(llm, "ainvoke"):
        result = await llm.ainvoke(prompt)
    else:
        result = llm.invoke(prompt)
    return extract_content(result)


async def run_agent(request:str, model: str)->dict:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(get_model(model), tools)
            agent_response = await agent.ainvoke({"messages": request})
            interpreted = await interpret_agent_response(agent_response, request, model)
            return {"raw": agent_response, "answer": interpreted}



# Run the async function
if __name__ == "__main__":

    # Edit the model name here - run `ollama list` to see available models
    model = "llama3.1"

    # Write example
    # request = "Create a new node with the label 'Person' and the property 'name' set to 'John Doe'."
    
    # Read example
    request = "How many nodes are in the graph?"
    result = asyncio.run(run_agent(request, model))
    print(result)