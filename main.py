import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.tools import extract_asset_text
from ocr_types.agent_type import MultimodalOCRAgent

load_dotenv()

# DEFAULT_ASSET_PATH = Path("assets/bryan-adams-summer-of-69.mp4")
DEFAULT_ASSET_PATH = "assets/ocr.pptx"
# DEFAULT_ASSET_PATH = "https://www.youtube.com/watch?v=3eT464L1YRA&list=RD3eT464L1YRA"
DEFAULT_DIAGRAM_PATH = Path("state_graph_ocr.png")
OCR_AGENT = "ocr_agent"
OCR_TOOL = "ocr_tool"
FINAL = "final_node"
TOOLS = [extract_asset_text]


def create_llm_agent() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        model="gpt-4o",
        temperature=0,
    ).bind_tools(TOOLS)


llm_agent: Any | None = None


def get_llm_agent() -> ChatOpenAI:
    global llm_agent
    if llm_agent is None:
        llm_agent = create_llm_agent()
    return llm_agent


def ocr_agent(
    state: MultimodalOCRAgent,
    agent: ChatOpenAI | None = None,
) -> MultimodalOCRAgent:
    asset_path = state["asset_path"]

    system_instruction = f"""
You are a multimodal OCR agent. You will be given an asset path for a file such as
an image, PDF, audio file, or video. Your task is to extract text from the asset.

Use the `extract_asset_text` tool to:
1. Detect the file type.
2. Detect whether image text is handwritten or printed.
3. Extract the text.
4. Save the output as a txt file.

Asset path: {asset_path}

After the tool returns, summarize where the output was saved.
"""

    active_agent = agent or get_llm_agent()
    response = active_agent.invoke(
        [SystemMessage(content=system_instruction)] + state["messages"]
    )

    return {
        "messages": [response],
        "asset_path": asset_path,
    }


def final_node(state: MultimodalOCRAgent) -> MultimodalOCRAgent:
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "tool":
            data = json.loads(msg.content)
            return {
                "asset_path": state["asset_path"],
                "file_type": data.get("file_type"),
                "image_text_type": data.get("image_text_type"),
                "extracted_text": data.get("extracted_text"),
                "output_path": data.get("output_path"),
                "messages": [
                    HumanMessage(
                        content=f"Saved extracted text to {data.get('output_path')}"
                    )
                ],
            }

    return {
        "asset_path": state["asset_path"],
        "messages": state["messages"],
    }


def build_state_graph():
    builder = StateGraph(MultimodalOCRAgent)
    builder.add_node(OCR_AGENT, ocr_agent)
    builder.add_node(OCR_TOOL, ToolNode(TOOLS))
    builder.add_node(FINAL, final_node)
    builder.add_edge(START, OCR_AGENT)
    builder.add_conditional_edges(
        OCR_AGENT,
        tools_condition,
        {
            "tools": OCR_TOOL,
            END: END,
        },
    )
    builder.add_edge(OCR_TOOL, FINAL)
    builder.add_edge(FINAL, END)
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
