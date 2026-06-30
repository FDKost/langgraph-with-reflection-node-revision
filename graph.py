from langgraph.graph import StateGraph, END
from state import State
from dataclasses import replace
from langchain_openai import OpenAI
import json

def draft(state: State) -> State:
    """Generate an initial answer to the question."""
    llm = OpenAI(temperature=0)
    prompt = f"Answer the following question:\n\n{state.question}"
    answer = llm.invoke(prompt)
    return replace(state, answer=answer, round=1)

def reflect(state: State) -> State:
    """Evaluate the answer and produce a verdict and critique points."""
    llm = OpenAI(temperature=0)
    prompt = (
        f"You are a reviewer. Evaluate the following answer to the question: \"{state.question}\".\n"
        f"The answer is:\n\n{state.answer}\n\n"
        "Respond with a JSON object containing:\n"
        "{\n  \"verdict\": \"ok\" or \"needs_revision\",\n  \"critique\": [\"point1\", \"point2\", ...]\n}\n"
        "Only output the JSON."
    )
    response = llm.invoke(prompt)
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Fallback if the LLM output is not valid JSON
        data = {"verdict": "needs_revision", "critique": ["Unable to parse LLM response."]}

    verdict = data.get("verdict", "needs_revision")
    critique = data.get("critique", [])
    return replace(state, verdict=verdict, critique=critique)

def rewrite(state: State) -> State:
    """Rewrite the answer based on critique points."""
    if state.verdict != "needs_revision":
        return state
    llm = OpenAI(temperature=0)
    critique_text = "\n".join(f"- {c}" for c in state.critique)
    prompt = (
        f"You are an AI assistant. Based on the following critique points, rewrite the answer to the question: \"{state.question}\".\n"
        f"The current answer is:\n\n{state.answer}\n\n"
        f"Critique points:\n{critique_text}\n\n"
        "Provide the revised answer."
    )
    new_answer = llm.invoke(prompt)
    return replace(state, answer=new_answer, round=state.round + 1)

def end(state: State) -> str:
    """End node."""
    return END

def create_graph(max_rounds: int = 2):
    """Create and compile the LangGraph."""
    graph = StateGraph(State)
    graph.add_node("draft", draft)
    graph.add_node("reflect", reflect)
    graph.add_node("rewrite", rewrite)
    graph.add_node("end", end)

    graph.set_entry_point("draft")
    graph.add_conditional_edges("draft", lambda x: "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda x: "rewrite" if x.verdict == "needs_revision" else "end",
    )
    graph.add_conditional_edges(
        "rewrite",
        lambda x: "reflect" if x.round < max_rounds else "end",
    )

    return graph.compile()
