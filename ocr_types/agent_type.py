from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MultimodalOCRAgent(TypedDict):
    """Shared LangGraph state for the OCR agent workflow."""

    asset_path: str
    file_type: Optional[str]
    image_text_type: Optional[str]
    extracted_text: Optional[str]
    output_path: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
