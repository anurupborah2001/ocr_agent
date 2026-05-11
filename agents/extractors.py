import json
import mimetypes
import tempfile
from pathlib import Path

from agents.vision_llm import (
    classify_image_text_type_with_llm,
    extract_handwritten_image_with_llm,
)

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
TEXT_TYPES = {".txt", ".md", ".json", ".xml", ".html", ".csv", ".yaml", ".yml"}
PDF_TYPES = {".pdf"}
DOCX_TYPES = {".docx"}
XLSX_TYPES = {".xlsx", ".xls"}
PPTX_TYPES = {".pptx"}
AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def detect_file_type(asset_path: str) -> str:
    suffix = Path(asset_path).suffix.lower()

    if suffix in IMAGE_TYPES:
        return "image"
    if suffix in PDF_TYPES:
        return "pdf"
    if suffix in DOCX_TYPES:
        return "docx"
    if suffix in XLSX_TYPES:
        return "excel"
    if suffix in PPTX_TYPES:
        return "pptx"
    if suffix in AUDIO_TYPES:
        return "audio"
    if suffix in VIDEO_TYPES:
        return "video"
    if suffix in TEXT_TYPES:
        return "text"

    mime_type, _ = mimetypes.guess_type(asset_path)
    return mime_type or "unknown"


def extract_text_file(asset_path: str) -> str:
    import pandas as pd

    path = Path(asset_path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_string(index=False)

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, indent=2, ensure_ascii=False)

    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf(asset_path: str) -> str:
    import fitz
    import pdfplumber
    import pytesseract
    from PIL import Image

    text_parts: list[str] = []

    with pdfplumber.open(asset_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(f"\n--- PAGE {page_number} ---\n{text}")

    if "".join(text_parts).strip():
        return "\n".join(text_parts)

    doc = fitz.open(asset_path)
    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=300)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            pix.save(tmp.name)
            image = Image.open(tmp.name)
            text = pytesseract.image_to_string(image)

        text_parts.append(f"\n--- PAGE {page_number} OCR ---\n{text}")

    return "\n".join(text_parts)


def extract_printed_image(asset_path: str) -> str:
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(asset_path))


def extract_pdf_as_images(asset_path: str) -> str:
    """
    Render every PDF page as an image, classify page text type,
    then extract using the vision LLM for handwritten or mixed pages
    and Tesseract for printed pages.
    """
    import fitz

    print("Extracting PDF as images with LLM vision classification...")
    doc = fitz.open(asset_path)
    output: list[str] = []

    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=300)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            pix.save(tmp.name)

            page_type = classify_image_text_type_with_llm(tmp.name)
            print(f"Page {page_number} classified as: {page_type}")
            if page_type in {"handwritten_image", "mixed_image"}:
                text = extract_handwritten_image_with_llm(tmp.name)
            elif page_type == "printed_image":
                text = extract_printed_image(tmp.name)
            else:
                text = ""

            output.append(f"\n--- PAGE {page_number} | {page_type} ---\n{text}")

    return "\n".join(output)


def extract_docx(asset_path: str) -> str:
    from docx import Document

    document = Document(asset_path)
    paragraphs = [
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    ]

    tables: list[str] = []
    for table in document.tables:
        for row in table.rows:
            tables.append(" | ".join(cell.text for cell in row.cells))

    return "\n".join(paragraphs + tables)


def extract_excel(asset_path: str) -> str:
    import pandas as pd

    sheets = pd.read_excel(asset_path, sheet_name=None)
    output: list[str] = []

    for sheet_name, dataframe in sheets.items():
        output.append(f"\n=== SHEET: {sheet_name} ===\n")
        output.append(dataframe.to_string(index=False))

    return "\n".join(output)


def extract_pptx(asset_path: str) -> str:
    from pptx import Presentation

    presentation = Presentation(asset_path)
    output: list[str] = []

    for slide_index, slide in enumerate(presentation.slides, start=1):
        output.append(f"\n--- SLIDE {slide_index} ---\n")

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                output.append(shape.text)

    return "\n".join(output)


def extract_audio(asset_path: str) -> str:
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(asset_path)
    return result["text"]


def extract_video(asset_path: str) -> str:
    import pytesseract
    from moviepy import VideoFileClip
    from PIL import Image

    output: list[str] = []

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_file:
        clip = VideoFileClip(asset_path)

        if clip.audio:
            clip.audio.write_audiofile(audio_file.name, logger=None)
            output.append("--- AUDIO TRANSCRIPTION ---")
            output.append(extract_audio(audio_file.name))

        output.append("\n--- FRAME OCR ---")

        duration = int(clip.duration)
        for second in range(0, duration, 5):
            frame = clip.get_frame(second)
            image = Image.fromarray(frame)
            text = pytesseract.image_to_string(image)

            if text.strip():
                output.append(f"\n--- FRAME {second}s ---\n{text}")

        clip.close()

    return "\n".join(output)


def extract_any(
    asset_path: str,
    file_type: str,
    image_text_type: str | None = None,
) -> str:
    if file_type == "text":
        return extract_text_file(asset_path)

    if file_type == "image":
        if image_text_type in {"handwritten_image", "mixed_image"}:
            return extract_handwritten_image_with_llm(asset_path)
        if image_text_type == "no_text_image":
            return ""
        return extract_printed_image(asset_path)

    if file_type == "pdf":
        return extract_pdf_as_images(asset_path)

    if file_type == "docx":
        return extract_docx(asset_path)

    if file_type == "excel":
        return extract_excel(asset_path)

    if file_type == "pptx":
        return extract_pptx(asset_path)

    if file_type == "audio":
        return extract_audio(asset_path)

    if file_type == "video":
        return extract_video(asset_path)

    raise ValueError(f"Unsupported file type: {file_type}")
