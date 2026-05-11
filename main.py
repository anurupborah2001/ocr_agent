import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import extract_text
from ocr_types.agent_type import OCRAgent

load_dotenv()

DEFAULT_ASSET_PATH = Path("assets/cursive_writing.pdf")
DEFAULT_DIAGRAM_PATH = Path("state_graph_ocr.png")
OCR_AGENT = "ocr_agent"
OCR_TOOL = "ocr_tool"
TOOLS = [extract_text]


def create_llm_agent() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        model="gpt-4o",
        temperature=0,
    ).bind_tools(TOOLS)


llm_agent = create_llm_agent()


def ocr_agent(state: OCRAgent, agent: ChatOpenAI | None = None) -> OCRAgent:
    asset_path = state["asset_path"]

    system_instruction = f"""
You are an OCR agent.

You can extract text from images, PDFs, and scanned documents.
Use the available OCR tool when text extraction is required.

Asset path:
{asset_path}
"""

    active_agent = agent or llm_agent
    response = active_agent.invoke(
        [SystemMessage(content=system_instruction)] + state["messages"]
    )

    return {
        "messages": [response],
        "asset_path": asset_path,
    }


def build_state_graph():
    builder = StateGraph(OCRAgent)
    builder.add_node(OCR_AGENT, ocr_agent)
    builder.add_node(OCR_TOOL, ToolNode(TOOLS))
    builder.add_edge(START, OCR_AGENT)
    builder.add_conditional_edges(
        OCR_AGENT,
        tools_condition,
        {
            "tools": OCR_TOOL,
            END: END,
        },
    )
    builder.add_edge(OCR_TOOL, OCR_AGENT)
    return builder.compile()


state_graph = build_state_graph()


def write_graph_diagram(
    graph=state_graph,
    output_path: Path = DEFAULT_DIAGRAM_PATH,
) -> str:
    compiled_graph = graph.get_graph()
    mermaid_diagram = compiled_graph.draw_mermaid()
    compiled_graph.draw_mermaid_png(output_file_path=str(output_path))
    return mermaid_diagram


def run_ocr(
    asset_path: str | Path = DEFAULT_ASSET_PATH,
    user_prompt: str = "Transcribe the provided asset and extract the text content.",
    graph=state_graph,
):
    normalized_asset_path = str(asset_path)
    return graph.invoke(
        {
            "asset_path": normalized_asset_path,
            "messages": [HumanMessage(content=user_prompt)],
        }
    )


if __name__ == "__main__":
    print(write_graph_diagram())
    print(run_ocr())
