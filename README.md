# LangGraph with Reflection Loop – Revision

## Overview

This project demonstrates a simple **reflection loop** built with LangGraph and LangChain.  
The agent:

1. Generates an initial answer to a user‑supplied question.
2. Uses an LLM to critique the answer and decide whether it is acceptable.
3. If the answer is not acceptable and the maximum number of rounds has not been reached, the agent rewrites the answer based on the critique.
4. The loop repeats until the answer is deemed acceptable or the maximum number of rounds is exceeded.

The loop is implemented using a `StateGraph` that manages the state of the conversation.

## Requirements

- Python 3.10+
- `langchain-openai`
- `langgraph`
- `openai` (for the LLM)

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py "Explain to a student the difference between tool and resource in MCP" --max-rounds 2
```

Optional arguments:

- `--max-rounds`: Maximum number of revision rounds (default: 2)
- `--model`: OpenAI model to use (default: `gpt-3.5-turbo`)
- `--temperature`: Temperature for the LLM (default: 0.7)

The script prints the final draft, critique, and verdict.

## Graph Logic

```
START → draft_answer → critique
          │
          ├─ if verdict == "ok" → END
          └─ if verdict == "needs_revision" and round < max_rounds → rewrite → critique
          └─ otherwise → END
```

The graph is implemented in `src/graph.py` using LangGraph’s `StateGraph`.  
The `ReflectState` TypedDict defines the state shape.

## Testing

Run the unit tests with:

```bash
pytest
```

The tests cover node outputs and the overall graph flow.

## License

MIT
