from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OCRAgent(TypedDict):
    """Shared LangGraph state for the OCR agent workflow."""

    asset_path: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
