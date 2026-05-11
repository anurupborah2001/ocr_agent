import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool

from agents.extractors import (
    classify_image_text_type,
    detect_file_type,
    extract_any,
    infer_output_stem,
    is_url,
)

load_dotenv()

logger = logging.getLogger(__name__)

OUTPUT_FOLDER = "output"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


@tool
def extract_asset_text(asset_path: str) -> str:
    """Detect the asset type, extract text, and persist the result as a txt file."""

    if is_url(asset_path):
        normalized_asset_path = asset_path
        file_type = detect_file_type(asset_path)
    else:
        asset = Path(asset_path).expanduser().resolve()

        if not asset.exists():
            raise FileNotFoundError(f"Asset file does not exist: {asset}")

        if not asset.is_file():
            raise ValueError(f"Asset path is not a file: {asset}")

        if asset.stat().st_size == 0:
            raise ValueError(f"Asset file is empty: {asset}")

        normalized_asset_path = str(asset)
        file_type = detect_file_type(normalized_asset_path)

    image_text_type = None
    if file_type == "image":
        image_text_type = classify_image_text_type(normalized_asset_path)

    extracted_text = extract_any(
        asset_path=normalized_asset_path,
        file_type=file_type,
        image_text_type=image_text_type,
    )

    output_path = (
        Path(OUTPUT_FOLDER) / f"{infer_output_stem(normalized_asset_path)}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(extracted_text or "", encoding="utf-8")

    logger.info("Saved OCR text to: %s", output_path)

    return json.dumps(
        {
            "asset_path": normalized_asset_path,
            "file_type": file_type,
            "image_text_type": image_text_type,
            "output_path": str(output_path),
            "extracted_text": extracted_text,
        },
        ensure_ascii=False,
    )
