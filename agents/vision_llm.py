import base64
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

llm_vision: Any | None = None


def get_llm_vision() -> ChatOpenAI:
    global llm_vision
    if llm_vision is None:
        llm_vision = ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
            model="gpt-4o",
            temperature=0,
        )
    return llm_vision


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)

    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "image/png")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def classify_image_text_type_with_llm(image_path: str) -> str:
    response = get_llm_vision().invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Classify the text type in this image.\n\n"
                            "Return only valid JSON:\n"
                            "{\n"
                            '  "type": "handwritten_image | printed_image | '
                            'mixed_image | no_text_image",\n'
                            '  "confidence": 0.0,\n'
                            '  "reason": "short reason"\n'
                            "}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                ]
            )
        ]
    )

    try:
        result = json.loads(response.content)
        print("LLM vision classification result:", result)
        return result.get("type", "printed_image")
    except Exception:
        return "printed_image"


def extract_handwritten_image_with_llm(image_path: str) -> str:
    response = get_llm_vision().invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Carefully transcribe all handwritten text in this image. "
                            "Return only the extracted text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                ]
            )
        ]
    )

    return response.content.strip()
