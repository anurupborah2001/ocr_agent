import importlib
import sys
from pathlib import Path
from types import ModuleType

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class StubAgent:
    def __init__(self, response_content: str = "Transcribed text"):
        self.response_content = response_content
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return AIMessage(content=self.response_content)


class StubCompiledGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"messages": payload["messages"], "asset_path": payload["asset_path"]}


class StubDrawableGraph:
    def __init__(self):
        self.output_path = None

    def draw_mermaid(self):
        return "graph TD; START-->OCR_AGENT"

    def draw_mermaid_png(self, output_file_path: str):
        self.output_path = output_file_path


class StubGraphWrapper:
    def __init__(self):
        self.drawable = StubDrawableGraph()

    def get_graph(self):
        return self.drawable


def load_main_with_langgraph_stubs(monkeypatch):
    stub_graph_module = ModuleType("langgraph.graph")
    stub_graph_message_module = ModuleType("langgraph.graph.message")
    stub_prebuilt_module = ModuleType("langgraph.prebuilt")

    class FakeStateGraph:
        def __init__(self, _state_type):
            self.nodes = []
            self.edges = []
            self.conditional_edges = []

        def add_node(self, name, node):
            self.nodes.append((name, node))

        def add_edge(self, source, target):
            self.edges.append((source, target))

        def add_conditional_edges(self, source, condition, mapping):
            self.conditional_edges.append((source, condition, mapping))

        def compile(self):
            return StubGraphWrapper()

    stub_graph_module.START = "START"
    stub_graph_module.END = "END"
    stub_graph_module.StateGraph = FakeStateGraph
    stub_graph_message_module.add_messages = lambda messages, new_messages: (
        messages + new_messages
    )
    stub_prebuilt_module.ToolNode = lambda tools: {"tools": tools}
    stub_prebuilt_module.tools_condition = object()

    monkeypatch.setitem(sys.modules, "langgraph.graph", stub_graph_module)
    monkeypatch.setitem(
        sys.modules, "langgraph.graph.message", stub_graph_message_module
    )
    monkeypatch.setitem(sys.modules, "langgraph.prebuilt", stub_prebuilt_module)
    sys.modules.pop("ocr_types.agent_type", None)
    sys.modules.pop("main", None)

    return importlib.import_module("main")


def test_ocr_agent_builds_system_prompt_with_asset_path(monkeypatch):
    main = load_main_with_langgraph_stubs(monkeypatch)
    stub_agent = StubAgent()
    state = {
        "asset_path": "assets/hand-written.png",
        "messages": [HumanMessage(content="Extract text from this file.")],
    }

    result = main.ocr_agent(state, agent=stub_agent)

    assert result["asset_path"] == "assets/hand-written.png"
    assert result["messages"][0].content == "Transcribed text"

    invocation = stub_agent.invocations[0]
    assert isinstance(invocation[0], SystemMessage)
    assert "assets/hand-written.png" in invocation[0].content
    assert invocation[1].content == "Extract text from this file."


def test_write_graph_diagram_writes_png_and_returns_mermaid(
    monkeypatch, tmp_path: Path
):
    main = load_main_with_langgraph_stubs(monkeypatch)
    graph_wrapper = StubGraphWrapper()
    output_path = tmp_path / "state_graph.png"

    mermaid = main.write_graph_diagram(graph=graph_wrapper, output_path=output_path)

    assert mermaid == "graph TD; START-->OCR_AGENT"
    assert graph_wrapper.drawable.output_path == str(output_path)


def test_run_ocr_invokes_graph_with_expected_payload(monkeypatch):
    main = load_main_with_langgraph_stubs(monkeypatch)
    graph = StubCompiledGraph()

    result = main.run_ocr(
        asset_path="assets/test.png",
        user_prompt="Read everything in this image.",
        graph=graph,
    )

    assert result["asset_path"] == "assets/test.png"
    assert len(graph.calls) == 1
    assert isinstance(graph.calls[0]["messages"][0], HumanMessage)
    assert graph.calls[0]["messages"][0].content == "Read everything in this image."
