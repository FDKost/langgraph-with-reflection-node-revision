import argparse
import os
from dotenv import load_dotenv
from graph import create_graph

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="LangGraph with Reflection Demo")
    parser.add_argument("question", type=str, help="The question to ask the AI.")
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=2,
        help="Maximum number of rewrite rounds (default: 2).",
    )
    args = parser.parse_args()

    # Create the graph with the specified max_rounds
    graph = create_graph(max_rounds=args.max_rounds)

    # Initial state
    state = {"question": args.question}

    # Run the graph
    final_state = graph.invoke(state)

    # Output results
    print("\n=== Final Answer ===")
    print(final_state["answer"])
    print("\n=== Verdict ===")
    print(final_state["verdict"])
    if final_state["critique"]:
        print("\n=== Critique Points ===")
        for idx, point in enumerate(final_state["critique"], 1):
            print(f"{idx}. {point}")

if __name__ == "__main__":
    main()
