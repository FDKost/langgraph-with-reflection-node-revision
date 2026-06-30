import pytest
from langchain_openai import ChatOpenAI
from src.graph import create_graph, ReflectState

@pytest.fixture
def mock_llm():
    class MockLLM:
        def invoke(self, prompt: str) -> str:
            if "Answer the following question" in prompt:
                return "Paris"
            if "Critique it" in prompt:
                return '{"verdict":"needs_revision","critique":"Missing detail"}'
            if "Rewrite the draft answer" in prompt:
                return "Paris, France"
            return ""
    return MockLLM()

def test_graph_flow(mock_llm):
    graph = create_graph(mock_llm)
    initial_state = {
        "question": "What is the capital of France?",
        "draft": "",
        "critique": "",
        "verdict": "",
        "round": 0,
        "max_rounds": 2,
    }
    final_state = graph.invoke(initial_state)
    assert final_state["draft"] == "Paris, France"
    assert final_state["critique"] == "Missing detail"
    assert final_state["verdict"] == "needs_revision"
