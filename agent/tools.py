import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

llm_ocr_agent = ChatOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
    model="gpt-4o",
    temperature=0,
)


@tool
def extract_text(asset_path: str) -> str:
    """
    Extract text from an image or PDF and save the extracted text
    as a .txt file using the same asset filename.
    """

    try:
        asset = Path(asset_path)

        with asset.open("rb") as file_handle:
            asset_bytes = file_handle.read()

        asset_base64 = base64.b64encode(asset_bytes).decode("utf-8")
        suffix = asset.suffix.lower()

        mime_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }
        mime_type = mime_type_map.get(suffix, "application/octet-stream")

        prompt = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this file. "
                            "Return only the extracted text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{asset_base64}"},
                    },
                ]
            )
        ]

        response = llm_ocr_agent.invoke(prompt)
        extracted_text = response.content.strip()
        output_file = Path(OUTPUT_FOLDER) / f"{asset.stem}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as file_handle:
            file_handle.write(extracted_text)

        logger.info("Saved OCR text to: %s", output_file)
        return extracted_text
    except Exception as exc:  # pragma: no cover - exercised via return path
        logger.exception("Error extracting text: %s", exc)
        return f"OCR extraction failed: {exc}"
