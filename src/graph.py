from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import json

class ReflectState(TypedDict):
    question: str
    draft: str
    critique: str
    verdict: str  # "ok" or "needs_revision"
    round: int
    max_rounds: int

def create_graph(llm: ChatOpenAI) -> StateGraph:
    graph = StateGraph(ReflectState)

    @graph.node
    def draft_answer(state: ReflectState) -> ReflectState:
        answer = llm.invoke(f"Answer the following question: {state['question']}")
        state["draft"] = answer
        state["round"] = 1
        return state

    @graph.node
    def critique(state: ReflectState) -> ReflectState:
        prompt = (
            f"Given the draft answer: {state['draft']}\n"
            f"Round: {state['round']}\n"
            "Critique it and decide if it is ok. Return JSON with keys "
            "\"verdict\" (\"ok\" or \"needs_revision\") and \"critique\"."
        )
        attempts = 0
        while attempts < state["max_rounds"]:
            try:
                result = llm.invoke(prompt)
                parsed = json.loads(result)
                state["verdict"] = parsed["verdict"]
                state["critique"] = parsed["critique"]
                return state
            except Exception as e:
                attempts += 1
                print(f"Critique attempt {attempts} failed: {e}")
        # If we exit the loop, set a default verdict
        state["verdict"] = "needs_revision"
        state["critique"] = f"Failed to generate critique after {attempts} attempts."
        return state

    @graph.node
    def rewrite(state: ReflectState) -> ReflectState:
        prompt = (
            f"Rewrite the draft answer: {state['draft']} based on the following critique: "
            f"{state['critique']}. Provide the revised answer. Round: {state['round']}"
        )
        new_draft = llm.invoke(prompt)
        state["draft"] = new_draft
        state["round"] += 1
        return state

    graph.add_node("draft_answer", draft_answer)
    graph.add_node("critique", critique)
    graph.add_node("rewrite", rewrite)

    graph.add_edge(START, "draft_answer")
    graph.add_edge("draft_answer", "critique")
    graph.add_edge("rewrite", "critique")

    def critique_condition(state: ReflectState) -> Literal["ok", "needs_revision_and_rounds_left", "needs_revision_and_no_rounds_left"]:
        if state["verdict"] == "ok":
            return "ok"
        elif state["verdict"] == "needs_revision" and state["round"] < state["max_rounds"]:
            return "needs_revision_and_rounds_left"
        else:
            return "needs_revision_and_no_rounds_left"

    graph.add_conditional_edges(
        "critique",
        critique_condition,
        {
            "ok": END,
            "needs_revision_and_rounds_left": "rewrite",
            "needs_revision_and_no_rounds_left": END,
        },
    )

    return graph
