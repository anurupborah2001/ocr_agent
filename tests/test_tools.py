from pathlib import Path
from types import SimpleNamespace

from agent import tools


class StubOCRModel:
    def __init__(self, content: str):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.content)


def test_extract_text_saves_output_and_uses_expected_mime_type(
    monkeypatch, tmp_path: Path
):
    asset_path = tmp_path / "sample.png"
    asset_path.write_bytes(b"fake-image-bytes")

    output_dir = tmp_path / "output"
    stub_model = StubOCRModel("Detected text\n")

    monkeypatch.setattr(tools, "OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setattr(tools, "llm_ocr_agent", stub_model)

    extracted_text = tools.extract_text.func(str(asset_path))

    assert extracted_text == "Detected text"
    assert (output_dir / "sample.txt").read_text(encoding="utf-8") == "Detected text"

    message_content = stub_model.prompts[0][0].content
    assert message_content[0]["type"] == "text"
    assert message_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_extract_text_returns_error_message_for_missing_file():
    result = tools.extract_text.func("missing-file.png")

    assert result.startswith("OCR extraction failed:")
