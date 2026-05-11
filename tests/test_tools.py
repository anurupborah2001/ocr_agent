import json
from pathlib import Path

from agents import tools


def test_extract_asset_text_saves_output_and_returns_metadata(
    monkeypatch, tmp_path: Path
):
    asset_path = tmp_path / "sample.png"
    asset_path.write_bytes(b"fake-image-bytes")
    output_dir = tmp_path / "output"

    monkeypatch.setattr(tools, "OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setattr(tools, "detect_file_type", lambda _path: "image")
    monkeypatch.setattr(
        tools, "classify_image_text_type_with_llm", lambda _path: "printed_image"
    )
    monkeypatch.setattr(tools, "extract_any", lambda **_kwargs: "Detected text")

    payload = json.loads(tools.extract_asset_text.func(str(asset_path)))

    assert payload["file_type"] == "image"
    assert payload["image_text_type"] == "printed_image"
    assert payload["extracted_text"] == "Detected text"
    assert Path(payload["output_path"]).read_text(encoding="utf-8") == "Detected text"


def test_extract_asset_text_rejects_missing_files():
    try:
        tools.extract_asset_text.func("missing-file.png")
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError for a missing asset path")
