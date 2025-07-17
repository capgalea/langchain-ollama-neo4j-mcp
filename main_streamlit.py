import streamlit as st
from neo4j import GraphDatabase
from collections import defaultdict
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
MODEL_OPTIONS = ["llama3.2", "mistral", "qwen3"]

def get_neo4j_graph():
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes = {}
    node_labels = defaultdict(set)
    node_properties = {}  # node_id -> dict of properties
    edges = []
    with driver.session(database=database) as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
        for record in result:
            n = record["n"]
            m = record["m"]
            r = record["r"]
            n_id = n.element_id if hasattr(n, 'element_id') else str(n.id)
            m_id = m.element_id if hasattr(m, 'element_id') else str(m.id)
            n_labels = list(n.labels) if hasattr(n, 'labels') else []
            m_labels = list(m.labels) if hasattr(m, 'labels') else []
            nodes[n_id] = n.get("name", n_id)
            nodes[m_id] = m.get("name", m_id)
            node_labels[n_id].update(n_labels)
            node_labels[m_id].update(m_labels)
            node_properties[n_id] = dict(n.items())
            node_properties[m_id] = dict(m.items())
            edges.append((n_id, m_id, r.type))
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

def update_graph_from_neo4j(net):
    nodes, node_labels, node_properties, edges = get_neo4j_graph()
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
    host = os.environ.get("FASTAPI_HOST", os.environ.get("FASTAPI_HOST"))
    port = os.environ.get("FASTAPI_PORT", os.environ.get("FASTAPI_PORT"))
    return f"http://{host}:{port}"


def main():
    """Main function to run the Streamlit application."""
    # Set up the Streamlit interface
    st.set_page_config(layout="wide")
    st.title("LangChain + Ollama + Neo4j MCP Demo")
    
    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_model" not in st.session_state:
        st.session_state.last_model = MODEL_OPTIONS[0]

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

        # Chat input
        user_input = st.chat_input("Type your message and press Enter...")

        # Process user input
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            
            # Call the FastAPI endpoint
            try:
                response = requests.get(
                    f"{get_api_url()}/query",
                    params={"command": user_input, "model": selected_model},
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agent_response = result.get("result", "No response")
                else:
                    agent_response = f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                agent_response = f"Request failed: {str(e)}"
            
            st.session_state.chat_history.append(("agent", agent_response))

        # Display chat history using st.chat_message (most recent at top)
        for role, msg in reversed(st.session_state.chat_history):
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(msg)

    with col2:
        st.header("Graph Viewer")
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        net = update_graph_from_neo4j(net)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.write_html(tmp_file.name)
            html_content = open(tmp_file.name, "r").read()
            st.components.v1.html(html_content, height=550, scrolling=True)

if __name__ == "__main__":
    main()
