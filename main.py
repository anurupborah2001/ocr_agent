import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import extract_text
from ocr_types.agent_type import OCRAgent

load_dotenv()

tools = [extract_text]

llm_agent = ChatOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
    model="gpt-4o",
    temperature=0,
).bind_tools(tools)


def ocr_agent(state: OCRAgent):
    asset_path = state["asset_path"]

    system_instruction = f"""
You are an OCR agent.

You can extract text from images, PDFs, and scanned documents.
Use the available OCR tool when text extraction is required.

Asset path:
{asset_path}
"""

    response = llm_agent.invoke(
        [SystemMessage(content=system_instruction)] + state["messages"]
    )

    return {
        "messages": [response],
        "asset_path": asset_path,
    }


OCR_AGENT = "ocr_agent"
OCR_TOOL = "ocr_tool"

builder = StateGraph(OCRAgent)

builder.add_node(OCR_AGENT, ocr_agent)
builder.add_node(OCR_TOOL, ToolNode(tools))

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

state_graph = builder.compile()

print(state_graph.get_graph().draw_mermaid())
state_graph.get_graph().draw_mermaid_png(output_file_path="state_graph_ocr.png")


if __name__ == "__main__":
    asset_path = "assets/hand-written.png"
    user_prompt = "Transcribe the provided asset and extract the text content."

    result = state_graph.invoke({
        "asset_path": asset_path,
        "messages": [HumanMessage(content=user_prompt)],
    })

    print(result)