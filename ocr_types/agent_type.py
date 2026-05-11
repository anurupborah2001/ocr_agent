
from typing import Annotated, Optional, TypedDict
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

class OCRAgent(TypedDict):
    """
    OCRAgent is a type definition for an agent that can perform Optical Character Recognition (OCR) tasks. 
    It defines the structure and expected behavior of the agent, including its capabilities and interactions.
    """
    asset_path: Optional[str]
    messages:  Annotated[list[AnyMessage], add_messages]