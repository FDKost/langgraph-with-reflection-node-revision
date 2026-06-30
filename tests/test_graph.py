import importlib
import sys
import pytest
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

@pytest.fixture
def mock_llm_ok():
    class MockLLM:
        def invoke(self, prompt: str) -> str:
            if "Answer the following question" in prompt:
                return "Paris"
            if "Critique it" in prompt:
                return '{"verdict":"ok","critique":"Good"}'
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
    assert final_state["round"] == 2  # reached max rounds

def test_no_revision_needed(mock_llm_ok):
    graph = create_graph(mock_llm_ok)
    initial_state = {
        "question": "What is the capital of France?",
        "draft": "",
        "critique": "",
        "verdict": "",
        "round": 0,
        "max_rounds": 3,
    }
    final_state = graph.invoke(initial_state)
    assert final_state["draft"] == "Paris"
    assert final_state["critique"] == "Good"
    assert final_state["verdict"] == "ok"
    assert final_state["round"] == 1  # only initial draft

def test_cli_invocation(monkeypatch, capsys):
    class MockLLM:
        def invoke(self, prompt: str) -> str:
            if "Answer the following question" in prompt:
                return "Paris"
            if "Critique it" in prompt:
                return '{"verdict":"ok","critique":"Good"}'
            if "Rewrite the draft answer" in prompt:
                return "Paris, France"
            return ""

    # Patch ChatOpenAI to return our mock LLM
    monkeypatch.setattr('langchain_openai.ChatOpenAI', lambda *args, **kwargs: MockLLM())

    # Reload main after patching
    import main
    importlib.reload(main)

    sys.argv = ['main.py', 'What is the capital of France?']
    main.main()
    captured = capsys.readouterr()
    assert "Draft:" in captured.out
    assert "Paris" in captured.out
    assert "Critique:" in captured.out
    assert "Good" in captured.out
    assert "Verdict:" in captured.out
    assert "ok" in captured.out
