import pytest
from src.graph import create_graph, ReflectState
from langchain_openai import ChatOpenAI

class DummyLLM:
    """
    Dummy LLM that returns deterministic responses based on the prompt.
    """
    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self.call_count = 0

    def invoke(self, prompt: str) -> str:
        self.call_count += 1
        if "Answer the following question" in prompt:
            return "Initial answer"
        if "Critique it" in prompt:
            import re, json
            m = re.search(r"Round: (\d+)", prompt)
            round_num = int(m.group(1)) if m else 1
            if round_num < self.max_rounds:
                return json.dumps({"verdict": "needs_revision", "critique": "Needs more detail"})
            else:
                return json.dumps({"verdict": "ok", "critique": "All good"})
        if "Rewrite the draft answer" in prompt:
            return "Revised answer"
        return ""

def test_graph_produces_answer():
    llm = DummyLLM(max_rounds=2)
    graph = create_graph(llm)
    initial_state: ReflectState = {
        "question": "What is 2+2?",
        "draft": "",
        "critique": "",
        "verdict": "",
        "round": 0,
        "max_rounds": 2,
    }
    final_state = graph.invoke(initial_state)
    assert final_state["draft"] == "Revised answer"
    assert final_state["verdict"] == "ok"
    assert final_state["round"] == 2

def test_max_rounds_respected():
    llm = DummyLLM(max_rounds=1)
    graph = create_graph(llm)
    initial_state: ReflectState = {
        "question": "Explain gravity.",
        "draft": "",
        "critique": "",
        "verdict": "",
        "round": 0,
        "max_rounds": 1,
    }
    final_state = graph.invoke(initial_state)
    # Only one round of rewrite should happen
    assert final_state["round"] == 1
    assert final_state["verdict"] == "ok"

def test_no_rewrite_on_ok():
    llm = DummyLLM(max_rounds=3)
    # Modify DummyLLM to return ok on first critique
    original_invoke = llm.invoke
    def first_ok_invoke(prompt: str) -> str:
        if "Critique it" in prompt:
            import json
            return json.dumps({"verdict": "ok", "critique": "All good"})
        return original_invoke(prompt)
    llm.invoke = first_ok_invoke

    graph = create_graph(llm)
    initial_state: ReflectState = {
        "question": "What is the capital of France?",
        "draft": "",
        "critique": "",
        "verdict": "",
        "round": 0,
        "max_rounds": 3,
    }
    final_state = graph.invoke(initial_state)
    assert final_state["draft"] == "Initial answer"
    assert final_state["verdict"] == "ok"
    assert final_state["round"] == 1
