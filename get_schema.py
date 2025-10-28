import asyncio
import os
import json
from dotenv import load_dotenv
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def get_neo4j_schema():
    """Fetch the Neo4j database schema using MCP tools"""
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.2.4", "--transport", "stdio"],
        env=dict(os.environ)
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            # Find the schema tool
            schema_tool = None
            for tool in tools:
                if 'schema' in tool.name.lower():
                    schema_tool = tool
                    break
            
            if schema_tool:
                result = await schema_tool.ainvoke({})
                return result
            else:
                return None

def format_schema_for_prompt(schema_data):
    """Format schema data into a readable prompt for the LLM"""
    if not schema_data:
        return "Schema information unavailable."
    
    prompt = "Neo4j Database Schema:\n\n"
    
    for node_info in schema_data:
        label = node_info.get('label', 'Unknown')
        prompt += f"Node Type: {label}\n"
        
        # Attributes
        attrs = node_info.get('attributes', {})
        if attrs:
            prompt += "  Properties:\n"
            for attr_name, attr_type in attrs.items():
                prompt += f"    - {attr_name}: {attr_type}\n"
        
        # Relationships
        rels = node_info.get('relationships', {})
        if rels:
            prompt += "  Relationships:\n"
            for rel_type, target_node in rels.items():
                prompt += f"    - {rel_type} -> {target_node}\n"
        
        prompt += "\n"
    
    return prompt

def format_schema_for_display(schema_data):
    """Format schema data for display in Streamlit"""
    if not schema_data:
        return "Schema information unavailable."
    
    display = []
    for node_info in schema_data:
        label = node_info.get('label', 'Unknown')
        attrs = node_info.get('attributes', {})
        rels = node_info.get('relationships', {})
        
        section = f"**{label}**\n"
        if attrs:
            section += "- Properties: " + ", ".join(f"`{k}`" for k in attrs.keys()) + "\n"
        if rels:
            section += "- Relationships: " + ", ".join(f"`{k}` → {v}" for k, v in rels.items()) + "\n"
        
        display.append(section)
    
    return "\n".join(display)

if __name__ == "__main__":
    schema = asyncio.run(get_neo4j_schema())
    print(json.dumps(schema, indent=2))
