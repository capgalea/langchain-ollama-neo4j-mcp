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
1. Edit the model name in `main_simple.py` to an Ollama model you have available
2. Optionally edit the request in `main_simple.py` to a prompt you want to run
3. Run the script

```bash
uv run main_simple.py
```

### Using Multiple MCP servers
1. Edit the model name in `main_multi.py` to an Ollama model you have available
2. Optionally edit the request in `main_multi.py` to a prompt you want to run
3. Run the script

```bash
uv run main_multi.py
```

### Interactive CLI
1. Edit the model name in `main_interactive.py` to an Ollama model you have available
2. Run the script

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


## Testing
To try out various models against the same set of prompts, update the `EVALUATIONS` list in `test_multi.py` and run the test script.

```bash
uv run test_multi.py
```

## License
MIT License