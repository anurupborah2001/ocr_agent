# OCR Agent

An agentic, multimodal OCR pipeline built with LangGraph and LangChain for extracting text from images, PDFs, Office documents, audio, video, and handwritten content.

The project uses a LangGraph workflow to decide when to call OCR tooling, route the asset through the correct extractor, and persist the extracted text to disk for downstream use.

## Overview

`ocr-agent` is designed as a graph-based extraction workflow rather than a single OCR function. A reasoning node receives the user request and asset path, invokes a multimodal extraction tool when needed, and then returns a structured result containing the detected file type, extracted text, and output location.

This structure makes the project easy to extend for:

- richer document understanding workflows
- validation and post-processing steps
- multi-step extraction pipelines
- human review or approval nodes

## Architecture

The LangGraph workflow now contains three logical stages:

- `ocr_agent`: the reasoning node that decides when to call the extraction tool
- `ocr_tool`: the tool execution node that performs file detection and extraction
- `final_node`: a post-processing node that parses the tool payload and writes the final fields back into graph state

The flow starts at `ocr_agent`. If the model decides the tool is needed, LangGraph routes execution to `ocr_tool`. After tool execution, control returns to `ocr_agent`, and the workflow eventually exits through `final_node`, which normalizes the extracted result for the final state.

### LangGraph Diagram

![OCR LangGraph Architecture](./state_graph_ocr.png)

## OCR Process

The current implementation follows this flow:

1. [main.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/main.py) builds a `StateGraph` using the typed state in [ocr_types/agent_type.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/ocr_types/agent_type.py).
2. The graph state tracks:
   - `asset_path`
   - `file_type`
   - `image_text_type`
   - `extracted_text`
   - `output_path`
   - `messages`
3. The `ocr_agent` node receives the current state and prompts the LLM to use `extract_asset_text` when OCR or transcription is required.
4. The tool in [agents/tools.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/agents/tools.py):
   - validates the input path
   - detects the high-level file type with `detect_file_type`
   - classifies image text as printed, handwritten, mixed, or no-text when applicable
   - dispatches to the appropriate extractor
   - saves the extracted text to `output/<asset-name>.txt`
   - returns a JSON payload with the extraction metadata
5. [agents/extractors.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/agents/extractors.py) routes extraction by media type:
   - text files are read directly
   - images use Tesseract or the vision LLM depending on text type
   - PDFs can be rendered page by page and processed as images
   - Word, Excel, and PowerPoint files are parsed with document libraries
   - audio uses Whisper transcription
   - video combines audio transcription with periodic frame OCR
6. [agents/vision_llm.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/agents/vision_llm.py) handles image classification and handwritten text transcription with a vision-capable `gpt-4o` model.
7. `final_node` parses the tool JSON and stores the final fields back into graph state so callers receive a structured result instead of only raw tool output.

## Project Structure

```text
ocr-agent/
├── agents/
│   ├── extractors.py         # File-type routing and media-specific extraction logic
│   ├── tools.py              # LangChain tool wrapper for extraction
│   └── vision_llm.py         # Vision model helpers for classification and handwriting OCR
├── assets/
│   └── hand-written.png      # Sample input asset
├── ocr_types/
│   └── agent_type.py         # Shared LangGraph state definition
├── output/
│   └── hand-written.txt      # Sample OCR output
├── tests/
│   ├── test_main.py          # Graph and orchestration tests
│   └── test_tools.py         # Tool-level tests
├── main.py                   # Graph assembly and execution entrypoint
├── pyproject.toml            # Dependencies and tool configuration
├── .pre-commit-config.yaml   # Pre-commit hooks
└── state_graph_ocr.png       # Generated LangGraph architecture diagram
```

## Supported Inputs

The extraction pipeline currently supports:

- images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`
- PDFs: `.pdf`
- text and data files: `.txt`, `.md`, `.json`, `.xml`, `.html`, `.csv`, `.yaml`, `.yml`
- Word documents: `.docx`
- spreadsheets: `.xlsx`, `.xls`
- PowerPoint: `.pptx`
- audio: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`
- video: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- LangChain OpenAI
- `gpt-4o`
- Tesseract OCR
- Whisper
- PyMuPDF
- `python-dotenv`

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

The current implementation uses GitHub-hosted or Azure-hosted inference endpoints depending on the model helper being called, and `GITHUB_TOKEN` is used as the API credential for the `ChatOpenAI` clients configured in code.

### 3. Run the workflow

```bash
python main.py
```

By default, the script:

- generates the Mermaid graph diagram
- exports `state_graph_ocr.png`
- runs the OCR workflow against the sample asset
- saves the extracted text under `output/`

## Testing And Quality Gates

Run the test suite directly with:

```bash
pytest
```

Install the git hooks with:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

Run all configured checks manually with:

```bash
pre-commit run --all-files
```

The current pre-commit setup includes:

- `pre-commit-hooks`
- `ruff`
- `black`
- conventional commit validation
- `pytest`

`pytest` is configured to run both on `pre-push` and when you execute `pre-commit run --all-files`.

## Design Highlights

- Multimodal extraction pipeline, not just image OCR
- Clear separation between graph orchestration, tool execution, file-type routing, and vision helpers
- Structured graph state for downstream consumers
- Persisted text output for follow-on workflows
- Test and pre-commit coverage for local quality checks

## Notes

- The sample execution path is defined in [main.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ocr-agent/main.py).
- Some extractors require local native/runtime dependencies such as Tesseract and multimedia/document libraries.
- Heavy dependencies are imported lazily inside extractor functions so unrelated workflows do not fail at import time.
- The graph diagram is generated programmatically when the script runs.
