# LangGraph with Reflection Node – Revision

## Requirements
- [high] Unify to Graph Paradigm: Remove any chain or try/except logic from the current implementation and refactor the entire workflow to use LangGraph’s graph API only.
- [high] Define ReflectState: Create a TypedDict named ReflectState with fields: question (str), draft (str), critique (str), verdict (str – "ok" or "needs_revision"), round (int), max_rounds (int, default 2).
- [high] Implement Nodes: Implement three nodes:
- draft_answer: generates the initial answer.
- reflect: uses an LLM to produce a verdict and 2–3 critique points; must not use exception handling.
- rewrite: updates the draft based on critique and increments round.
- [high] Configure Graph Flow: Set up the graph so that START → draft_answer → reflect, then:
- if verdict == "ok" → END;
- if verdict == "needs_revision" and round < max_rounds → rewrite → reflect;
- otherwise → END.
- [high] Enforce max_rounds: Ensure the agent stops after reaching max_rounds even if the verdict is still "needs_revision".
- [high] CLI Interface: Provide a command‑line interface that accepts a question, runs the graph, and prints the final draft, critique, and verdict.
- [normal] README and Documentation: Add a README explaining dependencies, how to run the CLI, and a brief description of the graph logic.
- [low] Testing and Linting: Include basic unit tests for node outputs and graph flow, and ensure code passes flake8/black formatting.
