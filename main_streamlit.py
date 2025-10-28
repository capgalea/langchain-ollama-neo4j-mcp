import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from collections import defaultdict
from pydantic import SecretStr
import requests
import asyncio
import os

# Load environment vars from .env
from dotenv import load_dotenv
load_dotenv()

# For graph visualization
from pyvis.network import Network
import tempfile


# Update options from `ollama list` here
# Supports Ollama (local), OpenAI (GPT), and Anthropic (Claude) models
MODEL_OPTIONS = [
    # Ollama local models
    "qwen2.5:1.5b",
    # OpenAI models (requires OPENAI_API_KEY)
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
    # Anthropic Claude models (requires ANTHROPIC_API_KEY)
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514",
]

def load_llm_api_keys():
    # ...existing code...
    openai_key = st.secrets.get("openai_api_key") or st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
    anthropic_key = st.secrets.get("anthropic_api_key") or st.secrets.get("anthropic", {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    return {
        "openai": SecretStr(openai_key or ""),
        "anthropic": SecretStr(anthropic_key or "")
    }

def get_neo4j_graph(cypher_query=None, limit=100):
    """
    Get graph data from Neo4j. If cypher_query is provided, visualize those results.
    Otherwise, show a random sample of the graph.
    """
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    if not uri or not user or not password:
        raise ValueError("Neo4j connection parameters (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) must be set in environment variables")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes = {}
    node_labels = defaultdict(set)
    node_properties = {}  # node_id -> dict of properties
    edges = []
    
    with driver.session(database=database) as session:
        # Use provided query or default to random sample
        if cypher_query:
            try:
                result = session.run(cypher_query)
            except Exception as e:
                # If query fails, fall back to default
                print(f"Query failed: {e}")
                result = session.run(f"MATCH (n)-[r]->(m) RETURN n, r, m LIMIT {limit}")
        else:
            result = session.run(f"MATCH (n)-[r]->(m) RETURN n, r, m LIMIT {limit}")
        
        for record in result:
            # Handle different return patterns
            for key in record.keys():
                value = record[key]
                
                # Handle nodes
                if hasattr(value, 'labels'):  # It's a node
                    node_id = value.element_id if hasattr(value, 'element_id') else str(value.id)
                    node_label = list(value.labels)[0] if value.labels else "Unknown"
                    
                    # Get display name from common properties
                    display_name = (value.get("name") or value.get("CI_Name") or 
                                  value.get("Grant_Title") or value.get("Admin_Institute") or 
                                  value.get("Field_of_Research") or value.get("Grant_Type") or
                                  value.get("Funding_Body") or node_label)
                    
                    nodes[node_id] = display_name[:50]  # Truncate long names
                    node_labels[node_id].update(value.labels)
                    node_properties[node_id] = dict(value.items())
                
                # Handle relationships
                elif hasattr(value, 'type'):  # It's a relationship
                    start_id = value.start_node.element_id if hasattr(value.start_node, 'element_id') else str(value.start_node.id)
                    end_id = value.end_node.element_id if hasattr(value.end_node, 'element_id') else str(value.end_node.id)
                    edges.append((start_id, end_id, value.type))
        
        # Also check for explicit n, r, m pattern
        if 'n' in record.keys() and 'r' in record.keys() and 'm' in record.keys():
            pass  # Already handled above
    
    driver.close()
    return nodes, node_labels, node_properties, edges

def get_label_colors(label_set):
    # Dynamically generate a unique color for each label using HSL
    n = len(label_set)
    colors = {}
    for i, label in enumerate(sorted(label_set)):
        # Evenly space hues around the color wheel
        hue = int(360 * i / max(1, n))
        # Use full saturation and 60% lightness for vivid colors
        color = f"hsl({hue}, 80%, 60%)"
        colors[label] = color
    return colors

def update_graph_from_neo4j(net, cypher_query=None, limit=100):
    """
    Update graph visualization with Neo4j data.
    If cypher_query is provided, visualize those results.
    """
    nodes, node_labels, node_properties, edges = get_neo4j_graph(cypher_query, limit)
    
    if not nodes:
        # Return empty network if no data
        return net
    
    all_labels = set(l for labels in node_labels.values() for l in labels)
    label_colors = get_label_colors(all_labels)
    
    for node_id, label in nodes.items():
        labels = node_labels[node_id]
        color = label_colors[list(labels)[0]] if labels else "#CCCCCC"
        props = node_properties.get(node_id, {})
        # Display key-value pairs as plain text, one per line
        title_text = "\n".join(f"{k}: {v}" for k, v in props.items())
        net.add_node(node_id, label=label, color=color, title=title_text)
    
    for src, dst, rel in edges:
        net.add_edge(src, dst, label=rel)
    
    return net

# Async helper for Streamlit
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No event loop running
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)

@st.cache_resource(show_spinner=False)
def get_api_url():
    # You can make this configurable if needed
    host = os.environ.get("FASTAPI_HOST", "127.0.0.1")
    port = os.environ.get("FASTAPI_PORT", "8002")
    return f"http://{host}:{port}"

def extract_cypher_query(response_text):
    """
    Extract Cypher query from agent response.
    Looks for MATCH, CREATE, MERGE patterns.
    Returns the LAST query found (typically the successful one after retries).
    """
    import re
    import json
    
    # Strategy 1: Try to find the successful query result in ToolMessage content
    # Look for patterns like: ToolMessage(content='[{...}]', name='read_neo4j_cypher'
    # This indicates a successful query with results
    tool_result_pattern = r"ToolMessage\(content='\[.*?\]',\s*name='read_neo4j_cypher'"
    tool_results = list(re.finditer(tool_result_pattern, response_text, re.DOTALL))
    
    if tool_results:
        # We found successful query results
        # Now look backward from the last successful result to find its query
        last_result_pos = tool_results[-1].start()
        text_before_result = response_text[:last_result_pos]
        
        # Look for the query that produced this result
        query_pattern = r"'query':\s*\"(MATCH[^\"]+)\"(?=\s*[,}])"
        queries_before = list(re.finditer(query_pattern, text_before_result, re.DOTALL | re.IGNORECASE))
        
        if queries_before:
            # Get the last query before the successful result
            query = queries_before[-1].group(1).strip()
        else:
            # Fallback: just get the last query in the entire response
            all_queries = re.findall(query_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if all_queries:
                query = all_queries[-1].strip()
            else:
                return None
    else:
        # No successful results found, try to find any Cypher query
        # Look for Cypher patterns in code blocks or plain text
        patterns = [
            r'```cypher\s*(.*?)\s*```',  # Code block with cypher
            r'```\s*(MATCH.*?RETURN.*?)\s*```',    # Code block with MATCH...RETURN  
            r'(MATCH\s+.*?RETURN.*?)(?:\n\n|\Z)',  # MATCH...RETURN until double newline or end
        ]
        
        query = None
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                query = match.group(1).strip()
                break
        
        if not query:
            return None
    
    # Unescape the query - handle both single backslash and double backslash escaping
    # Remove line continuation backslashes (backslash at end of line)
    query = re.sub(r'\\\s*\n', '\n', query)  # Replace "\ \n" with just "\n"
    
    # Convert escaped characters back to normal
    query = query.replace('\\n', '\n')  # Unescape newlines
    query = query.replace('\\\'', "'")  # Unescape single quotes
    query = query.replace('\\"', '"')   # Unescape double quotes
    query = query.replace('\\t', '\t')  # Unescape tabs
    
    # Clean up the query - remove extra whitespace but preserve structure
    query = re.sub(r'[ \t]+', ' ', query)  # Normalize spaces/tabs
    query = re.sub(r'\n+', '\n', query)  # Normalize multiple newlines to single
    query = query.strip()
    
    # Validate it looks like a Cypher query
    if 'RETURN' in query.upper() and 'MATCH' in query.upper():
        return query
    
    return None

def get_query_results_as_dataframe(cypher_query, limit=100):
    """
    Execute a Cypher query and return results as a pandas DataFrame.
    Returns None if query fails or returns no tabular data.
    """
    import pandas as pd
    from neo4j import GraphDatabase
    
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        st.warning("⚠️ Neo4j credentials not found in environment variables.")
        return None
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            # Add LIMIT if not already present
            query = cypher_query
            if 'LIMIT' not in query.upper():
                query = f"{query} LIMIT {limit}"
            
            result = session.run(query)
            records = list(result)
            
            if not records:
                return pd.DataFrame()  # Return empty DataFrame instead of None
            
            # Convert to list of dicts
            data = []
            for record in records:
                row = {}
                for key in record.keys():
                    value = record[key]
                    
                    # Handle different Neo4j data types
                    if value is None:
                        row[key] = None
                    elif isinstance(value, (str, int, float, bool)):
                        # Simple types - use directly
                        row[key] = value
                    elif isinstance(value, list):
                        # Lists - convert to string representation
                        row[key] = str(value)
                    elif hasattr(value, '__class__'):
                        class_name = value.__class__.__name__
                        if class_name == 'Node':
                            # Extract all properties from Node
                            props = dict(value.items())
                            # Create a readable representation
                            label = list(value.labels)[0] if hasattr(value, 'labels') and value.labels else 'Node'
                            # Try to find the most meaningful property to display
                            display_val = None
                            for prop_key in ['Application_ID', 'Grant_Title', 'CI_Name', 'Admin_Institute', 'name', 'title']:
                                if prop_key in props:
                                    display_val = props[prop_key]
                                    break
                            if display_val:
                                row[key] = f"{label}: {display_val}"
                            else:
                                # Just show first few properties
                                row[key] = f"{label}: {dict(list(props.items())[:2])}"
                        elif class_name == 'Relationship':
                            # For relationships, show the type
                            row[key] = f"[{value.type}]"
                        else:
                            # For any other object, convert to string
                            row[key] = str(value)
                    else:
                        row[key] = str(value)
                        
                data.append(row)
            
            df = pd.DataFrame(data)
            driver.close()
            return df
            
    except Exception as e:
        st.error(f"❌ Error executing query: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None


def main():
    """Main function to run the Streamlit application."""
    import datetime
    import json
    
    # Set up the Streamlit interface
    st.set_page_config(layout="wide")
    st.title("LangChain + Ollama + Neo4j MCP Demo")
    
    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_model" not in st.session_state:
        st.session_state.last_model = MODEL_OPTIONS[0]
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "last_cypher_query" not in st.session_state:
        st.session_state.last_cypher_query = None
    if "visualize_results" not in st.session_state:
        st.session_state.visualize_results = False
    if "graph_html" not in st.session_state:
        st.session_state.graph_html = None
    if "prev_viz_mode" not in st.session_state:
        st.session_state.prev_viz_mode = None
    if "prev_viz_limit" not in st.session_state:
        st.session_state.prev_viz_limit = None

    col1, col2 = st.columns([1, 1])

    with col1:
        # Model selection dropdown
        selected_model = st.selectbox(
            "Choose LLM model:", 
            MODEL_OPTIONS, 
            index=MODEL_OPTIONS.index(st.session_state.last_model)
        )

        # Reset chat history if model changes
        if st.session_state.last_model != selected_model:
            st.session_state.chat_history = []
            st.session_state.last_model = selected_model

        # Database Schema Display
        with st.expander("📊 Database Schema", expanded=False):
            if "schema_data" not in st.session_state:
                try:
                    # Fetch schema from FastAPI backend
                    api_url = get_api_url()
                    response = requests.get(f"{api_url}/schema", timeout=30)
                    if response.status_code == 200:
                        schema_data = response.json()
                        # Handle if schema is returned as string
                        if isinstance(schema_data, str):
                            import json
                            schema_data = json.loads(schema_data)
                        st.session_state.schema_data = schema_data
                    else:
                        st.session_state.schema_data = None
                except Exception as e:
                    st.warning(f"Could not fetch schema: {e}")
                    st.session_state.schema_data = None
            
            if st.session_state.schema_data and isinstance(st.session_state.schema_data, list):
                for node_info in st.session_state.schema_data:
                    label = node_info.get('label', 'Unknown')
                    st.markdown(f"**Node: {label}**")
                    
                    attrs = node_info.get('attributes', {})
                    if attrs:
                        st.markdown("*Properties:*")
                        for k, v in attrs.items():
                            st.markdown(f"  - `{k}` ({v})")
                    
                    rels = node_info.get('relationships', {})
                    if rels:
                        st.markdown("*Relationships:*")
                        for k, v in rels.items():
                            st.markdown(f"  - `{k}` → {v}")
                    
                    st.markdown("---")
            else:
                st.info("Schema information not available")

        # Query History Display
        with st.expander("📜 Query History", expanded=False):
            if st.session_state.query_history:
                st.markdown(f"**Total Queries: {len(st.session_state.query_history)}**")
                
                # Add buttons in columns
                col_clear, col_export = st.columns(2)
                with col_clear:
                    if st.button("🗑️ Clear History"):
                        st.session_state.query_history = []
                        st.rerun()
                
                with col_export:
                    history_json = json.dumps(st.session_state.query_history, indent=2)
                    st.download_button(
                        label="📥 Export JSON",
                        data=history_json,
                        file_name=f"query_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                st.markdown("---")
                
                # Display queries in reverse order (most recent first)
                for idx, entry in enumerate(reversed(st.session_state.query_history)):
                    with st.container():
                        st.markdown(f"**Query #{len(st.session_state.query_history) - idx}** - *{entry['timestamp']}*")
                        st.markdown(f"**Model:** `{entry['model']}`")
                        st.markdown(f"**Question:** {entry['query']}")
                        
                        # Show response in a code block if it's short, or expander if long
                        if len(entry['response']) < 500:
                            st.markdown(f"**Answer:** {entry['response']}")
                        else:
                            with st.expander("Show Answer"):
                                st.markdown(entry['response'])
                        
                        if 'execution_time' in entry:
                            st.caption(f"⏱️ {entry['execution_time']:.2f}s")
                        
                        st.markdown("---")
            else:
                st.info("No queries yet. Start asking questions!")

        # Chat input
        user_input = st.chat_input("Type your message and press Enter...")

        # Process user input
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            
            # Call the FastAPI endpoint
            execution_time = 0
            cypher_query = None
            
            try:
                with st.spinner(f"Processing with {selected_model}... This may take a few minutes..."):
                    response = requests.get(
                        f"{get_api_url()}/query",
                        params={"command": user_input, "model": selected_model},
                        timeout=300  # Increased to 5 minutes
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    agent_response = result.get("result", "No response")
                    execution_time = result.get("seconds_to_complete", 0)
                    
                    # Try to extract Cypher query from the raw response
                    raw_response = result.get("raw", "")
                    cypher_query = extract_cypher_query(str(raw_response))
                    
                    # If not in raw, try in the result
                    if not cypher_query:
                        cypher_query = extract_cypher_query(agent_response)
                    
                    # Store for visualization
                    if cypher_query:
                        st.session_state.last_cypher_query = cypher_query
                        st.session_state.visualize_results = True
                else:
                    agent_response = f"Error: {response.status_code} - {response.text}"
            except requests.exceptions.Timeout:
                agent_response = f"Request timed out after 5 minutes. The query may be too complex or the model is taking too long. Try a simpler query or a faster model."
            except Exception as e:
                agent_response = f"Request failed: {str(e)}"
            
            st.session_state.chat_history.append(("agent", agent_response))
            
            # Add to query history
            st.session_state.query_history.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": selected_model,
                "query": user_input,
                "response": agent_response,
                "execution_time": execution_time,
                "cypher_query": cypher_query
            })

        # Display chat history using st.chat_message (most recent at top)
        for role, msg in reversed(st.session_state.chat_history):
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(msg)
        
        # Display interactive table if we have a recent query with Cypher
        if st.session_state.last_cypher_query:
            st.divider()
            st.subheader("📊 Query Results Table")
            
            # Show the extracted query for debugging
            with st.expander("🔍 Extracted Cypher Query", expanded=False):
                st.code(st.session_state.last_cypher_query, language="cypher")
            
            with st.spinner("Loading query results..."):
                df = get_query_results_as_dataframe(st.session_state.last_cypher_query, limit=1000)
                
                if df is not None and not df.empty:
                    st.success(f"✅ Found {len(df)} records with {len(df.columns)} columns")
                    
                    # Display the dataframe
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=400,
                        hide_index=False
                    )
                    
                    # Add download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"query_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif df is not None:
                    st.warning("⚠️ Query executed but returned no data.")
                else:
                    st.error("❌ Could not generate table. Check the Cypher query above for errors.")

    with col2:
        st.header("Graph Viewer")
        
        # Add controls for visualization
        col_viz1, col_viz2 = st.columns([3, 1])
        with col_viz1:
            # Determine default index based on whether we have a query
            default_index = 0 if (st.session_state.visualize_results and st.session_state.last_cypher_query) else 1
            
            viz_mode = st.radio(
                "Visualization Mode:",
                ["Query Results", "Random Sample"],
                index=default_index,
                horizontal=True,
                key="viz_mode_radio"
            )
        
        with col_viz2:
            viz_limit = st.number_input("Max Nodes:", min_value=10, max_value=500, value=100, step=10, key="viz_limit")
        
        # Show info if Query Results selected but no query available
        if viz_mode == "Query Results" and not st.session_state.last_cypher_query:
            st.info("💡 No query results yet. Ask a database question first, then switch to 'Query Results' mode.")
            viz_mode = "Random Sample"  # Fall back to random sample
        
        # Show Cypher query if available
        if st.session_state.last_cypher_query and viz_mode == "Query Results":
            with st.expander("🔍 Cypher Query", expanded=False):
                st.code(st.session_state.last_cypher_query, language="cypher")
        
        # Check if we need to regenerate the graph
        need_regenerate = (
            st.session_state.graph_html is None or
            st.session_state.prev_viz_mode != viz_mode or
            st.session_state.prev_viz_limit != viz_limit or
            (viz_mode == "Query Results" and st.session_state.visualize_results)
        )
        
        if need_regenerate:
            # Create network visualization
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color=True)
            
            try:
                if viz_mode == "Query Results" and st.session_state.last_cypher_query:
                    # Visualize the query results
                    st.caption(f"📊 Showing results from last query...")
                    net = update_graph_from_neo4j(net, st.session_state.last_cypher_query, viz_limit)
                    if not net.nodes:
                        st.info("Query returned no graph data. Showing random sample instead.")
                        net = update_graph_from_neo4j(net, None, viz_limit)
                else:
                    # Show random sample
                    st.caption(f"📊 Showing random sample of graph...")
                    net = update_graph_from_neo4j(net, None, viz_limit)
                
                # Generate and store HTML
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    net.write_html(tmp_file.name)
                    st.session_state.graph_html = open(tmp_file.name, "r").read()
                
                # Update tracking variables
                st.session_state.prev_viz_mode = viz_mode
                st.session_state.prev_viz_limit = viz_limit
                st.session_state.visualize_results = False  # Reset flag
                
            except Exception as e:
                st.error(f"Error visualizing graph: {str(e)}")
                st.info("Showing random sample instead.")
                net = Network(height="500px", width="100%", bgcolor="#222222", font_color=True)
                net = update_graph_from_neo4j(net, None, 50)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    net.write_html(tmp_file.name)
                    st.session_state.graph_html = open(tmp_file.name, "r").read()
                st.session_state.prev_viz_mode = viz_mode
                st.session_state.prev_viz_limit = viz_limit
        else:
            # Show caption based on current mode
            if viz_mode == "Query Results" and st.session_state.last_cypher_query:
                st.caption(f"📊 Showing results from last query...")
            else:
                st.caption(f"📊 Showing random sample of graph...")
        
        # Display the graph
        if st.session_state.graph_html:
            components.html(st.session_state.graph_html, height=550, scrolling=True)

if __name__ == "__main__":
    main()
