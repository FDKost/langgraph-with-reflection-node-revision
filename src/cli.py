import argparse
from langchain_openai import ChatOpenAI
from src.graph import create_graph

def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph Reflection Agent")
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument("--max-rounds", type=int, default=2, help="Maximum number of revision rounds")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the LLM")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
    )

    graph = create_graph(llm)

    initial_state = {
        "question": args.question,
        "draft": "",
        "critique": "",
        "verdict": "",
        "round": 0,
        "max_rounds": args.max_rounds,
    }

    final_state = graph.invoke(initial_state)

    print("\n=== Final Result ===")
    print(f"Draft:\n{final_state['draft']}\n")
    print(f"Critique:\n{final_state['critique']}\n")
    print(f"Verdict: {final_state['verdict']}\n")

if __name__ == "__main__":
    main()
