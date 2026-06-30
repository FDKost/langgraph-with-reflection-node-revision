# LangGraph with Reflection Demo

This project demonstrates a simple LangGraph workflow that generates an answer to a user‑supplied question, evaluates the answer with a reflection node, and rewrites the answer if necessary. The process repeats until the answer is deemed satisfactory or a maximum number of rounds is reached.

## Features

- **Draft node** – Generates an initial answer using OpenAI.
- **Reflection node** – Uses an LLM to evaluate the answer and produce a verdict (`ok` or `needs_revision`) along with critique points.
- **Rewrite node** – Rewrites the answer based on the critique and increments the round counter.
- **Max rounds logic** – Stops after a configurable number of rounds (default 2) even if the answer still needs revision.
- **Command‑line interface** – Run the graph from the terminal with a question argument.

## Prerequisites

- Python 3.10+
- An OpenAI API key. Set it in a `.env` file:

```dotenv
OPENAI_API_KEY=your-openai-key-here
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/langgraph-reflection-demo.git
cd langgraph-reflection-demo

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\\Scripts\\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py "Explain the theory of relativity in simple terms."
```

Optional arguments:

- `--max_rounds N` – Set the maximum number of rewrite rounds (default is 2).

Example:

```bash
python main.py "Explain the theory of relativity in simple terms." --max_rounds 3
```

The script will print the final answer, the verdict, and any critique points.

## Project Structure

```
├── graph.py          # LangGraph definition
├── state.py          # Dataclass for graph state
├── main.py           # CLI entry point
├── requirements.txt  # Python dependencies
└── README.md         # Documentation
```

## License

MIT License
