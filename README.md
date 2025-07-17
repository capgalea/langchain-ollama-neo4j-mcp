# Langchain + Ollama + Neo4j MCP

Simple examples of how to use Langchain, Ollama, and the Neo4j MCP server to interact with a Neo4j database. Including examples of chaining this base function for use in a FastAPI server and Streamlit app.

## Installation
1. Download or clone this repository
2. Install dependencies
    ```bash
    uv sync
    ```
3. Copy the `env.sample` file to `.env` and fill in credentials
    ```bash
    cp env.sample .env
    ```

## Setup
1. Start the target Neo4j server
2. Start ollama
3. Adjust any ollama model name references in the sample code


## Running

### Simple function
Run a single prompt against the target Neo4j database
```bash
uv run main_simple.py
```

### Using Multiple MCP servers
Run a single prompt against the target Neo4j database
```bash
uv run main_multi.py
```

### Interactive CLI
```bash
uv run main_interactive.py
```

### FastAPI option
```bash
 uv run main_fastapi.py
 ```

### Streamlit 
1. Run the earlier FastAPI option in separate terminal
2. Then start the streamlit app
```bash
uv run streamlit run main_streamlit.py
```


## License
MIT License