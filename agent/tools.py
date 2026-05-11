import os
import base64
import logging

from pathlib import Path

from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
    Extract text from image/PDF and save the extracted text
    as a .txt file using the same asset filename.
    """

    try:
        asset = Path(asset_path)

        # Read file as bytes
        with open(asset_path, "rb") as f:
            asset_bytes = f.read()

        # Convert to base64
        asset_base64 = base64.b64encode(asset_bytes).decode("utf-8")

        suffix = asset.suffix.lower()

        mime_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }

        mime_type = mime_type_map.get(
            suffix,
            "application/octet-stream"
        )

        # Vision prompt
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
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{asset_base64}"
                            )
                        },
                    },
                ]
            )
        ]

        response = llm_ocr_agent.invoke(prompt)
        extracted_text = response.content.strip()
        ##Save extracted text to .txt file with same name as asset
        output_file = (
            Path(OUTPUT_FOLDER)
            / f"{asset.stem}.txt"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        logger.info(
            f"Saved OCR text to: {output_file}"
        )

        return extracted_text

    except Exception as e:
        logger.exception(
            f"Error extracting text: {str(e)}"
        )
        return f"OCR extraction failed: {str(e)}"