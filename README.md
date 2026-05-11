# OCR Agent

An agentic, multimodal OCR and transcription pipeline built with LangGraph and LangChain for extracting text from images, PDFs, Excel workbooks, PowerPoint decks, audio, video, and supported remote URLs.

The project routes each asset through the right extractor, persists the extracted text to the `output/` directory, and returns structured metadata back through the graph.

## Overview

`ocr-agent` is implemented as a LangGraph workflow instead of a single extraction function. A reasoning node decides when to call the extraction tool, the tool detects the asset type and dispatches to the correct extractor, and a final node normalizes the tool result into graph state.

This architecture is designed for:

- multimodal OCR and transcription
- large-file extraction with structured output
- downstream automation and post-processing
- future review, summarization, or enrichment steps

## Architecture

The graph currently contains three main stages:

- `ocr_agent`: decides whether tool execution is required
- `ocr_tool`: runs the `extract_asset_text` tool
- `final_node`: parses the tool payload and writes structured fields back into state

### LangGraph Diagram

![OCR LangGraph Architecture](./state_graph_ocr.png)

## Extraction Flow

The current implementation works like this:

1. [main.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/main.py) builds a `StateGraph` using [ocr_types/agent_type.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/ocr_types/agent_type.py).
2. The graph state tracks:
   - `asset_path`
   - `file_type`
   - `image_text_type`
   - `extracted_text`
   - `output_path`
   - `messages`
3. [agents/tools.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/agents/tools.py) validates the asset, detects its type, calls the right extractor, and writes the result to `output/<asset-name>.txt`.
4. [agents/extractors.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/agents/extractors.py) handles media-specific extraction:
   - images: Tesseract or vision-LLM handwriting extraction
   - PDFs: direct text extraction or page-image OCR
   - Excel: row-chunked sheet extraction
   - PowerPoint: slide text extraction
   - audio: preprocessing with `ffmpeg` and Whisper transcription
   - video: audio extraction from the video, then reuse of `extract_audio()`
   - URLs: download/resolve remote assets before extraction where supported
5. `final_node` returns a structured result instead of sending the full tool output back through the LLM again, which avoids large payload failures.

## Project Structure

```text
ocr-agent/
├── agents/
│   ├── extractors.py
│   ├── tools.py
│   └── vision_llm.py
├── assets/
├── ocr_types/
├── output/
├── tests/
├── main.py
├── pyproject.toml
├── .pre-commit-config.yaml
└── state_graph_ocr.png
```

## Supported Inputs

- images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`
- PDFs: `.pdf`
- text and data files: `.txt`, `.md`, `.json`, `.xml`, `.html`, `.csv`, `.yaml`, `.yml`
- Word: `.docx`
- spreadsheets: `.xlsx`, `.xls`
- PowerPoint: `.pptx`
- audio: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`
- video: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- remote URLs for supported image, PDF, audio, and video flows

## Bundled Sample Assets

The repository currently includes these example inputs:

| Asset | Type | Output |
| --- | --- | --- |
| `assets/hand-written.png` | Handwritten image | `output/hand-written.txt` |
| `assets/cursive_writing.pdf` | PDF worksheet | `output/cursive_writing.txt` |
| `assets/audit-excel.xlsx` | Excel workbook | `output/audit-excel.txt` |
| `assets/ocr.pptx` | PowerPoint deck | `output/ocr.txt` |
| `assets/bryan-adams-cloud-9.mp3` | Audio | `output/bryan-adams-cloud-9.txt` |
| `assets/bryan-adams-summer-of-69.mp4` | Video | `output/bryan-adams-summer-of-69.txt` |

## Sample Output

Below are representative snippets from the actual generated files in `output/`.

### Handwritten Image

Asset: [hand-written.png](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/assets/hand-written.png)  
Output: [hand-written.txt](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/output/hand-written.txt)

```text
Lorem ifsum dolor sit amet, consectetur

adifiscing Chit. Morbi dolor Libero, rhoncus et
sapien vitae, voluthat cowallis lectus. Etiam
lobortis eget facus id maximus.
```

### PDF

Asset: [cursive_writing.pdf](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/assets/cursive_writing.pdf)  
Output: [cursive_writing.txt](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/output/cursive_writing.txt)

```text
--- PAGE 1 | printed_image ---
Kidde, Cursive Writing Practice

Worksheet

Name : Date :
```

### Excel

Asset: [audit-excel.xlsx](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/assets/audit-excel.xlsx)  
Output: [audit-excel.txt](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/output/audit-excel.txt)

```text
=== SHEET: Data ===

--- ROWS 1 TO 200 OF 500 ---
Transaction_ID       Date    Category        Account_Name Department
     TXN-10000 04/13/2025     Expense            Salaries        R&D
     TXN-10001 08/03/2025     Revenue     Consulting Fees Operations
```

### PowerPoint

Asset: [ocr.pptx](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/assets/ocr.pptx)  
Output: [ocr.txt](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/output/ocr.txt)

```text
--- SLIDE 1 ---

State of and Trends in OCR Technology
By Jim Hill, Solutions Specialist
```

### Audio

Asset: [bryan-adams-cloud-9.mp3](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/assets/bryan-adams-cloud-9.mp3)  
Output: [bryan-adams-cloud-9.txt](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/output/bryan-adams-cloud-9.txt)

```text
Clue number one was when you knocked on my door.
Clue number two was the look that you wore.
And that's when I knew it was a pretty good sign.
```

### Video

Asset: [bryan-adams-summer-of-69.mp4](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/assets/bryan-adams-summer-of-69.mp4)  
Output: [bryan-adams-summer-of-69.txt](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/output/bryan-adams-summer-of-69.txt)

```text
I got my first real six string
Bought it at the five and dime
Played it till my fingers bled
Was the summer of 69
```

## How To Run

### 1. Install dependencies

With `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .[dev]
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GITHUB_TOKEN=your_token_here
```

### 3. Install native/runtime requirements

For audio and video extraction on macOS:

```bash
brew install ffmpeg
```

### 4. Run the workflow

```bash
python main.py
```

By default, [main.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/main.py) currently points at an Excel sample asset, generates the graph diagram, runs extraction, and writes the result into `output/`.

## Testing And Quality Gates

Run tests directly with:

```bash
pytest
```

Install hooks:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

Run all checks:

```bash
pre-commit run --all-files
```

The configured checks include:

- `pre-commit-hooks`
- `ruff`
- `black`
- conventional commit validation
- `pytest`

## Design Highlights

- Multimodal extraction across document, image, audio, and video inputs
- URL-aware extraction for supported remote assets
- Excel extraction chunking to avoid oversized payloads
- Video transcription built by reusing the audio extractor
- Structured LangGraph state and persisted output files

## Notes

- Large workbook output is chunked by row range in the generated `.txt` file.
- Some extractors depend on local native/runtime tools such as `ffmpeg` and Tesseract.
- Heavy dependencies are imported lazily inside extractor functions where practical.
- The generated text quality depends on source quality, handwriting complexity, audio clarity, and model/backend availability.
