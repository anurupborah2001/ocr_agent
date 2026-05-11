import json
from pathlib import Path
from types import SimpleNamespace

from agents import extractors, tools


def test_extract_asset_text_saves_output_and_returns_metadata(
    monkeypatch, tmp_path: Path
):
    asset_path = tmp_path / "sample.png"
    asset_path.write_bytes(b"fake-image-bytes")
    output_dir = tmp_path / "output"

    monkeypatch.setattr(tools, "OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setattr(tools, "detect_file_type", lambda _path: "image")
    monkeypatch.setattr(
        tools, "classify_image_text_type", lambda _path: "printed_image"
    )
    monkeypatch.setattr(tools, "extract_any", lambda **_kwargs: "Detected text")

    payload = json.loads(tools.extract_asset_text.func(str(asset_path)))

    assert payload["file_type"] == "image"
    assert payload["image_text_type"] == "printed_image"
    assert payload["extracted_text"] == "Detected text"
    assert Path(payload["output_path"]).read_text(encoding="utf-8") == "Detected text"


def test_extract_asset_text_accepts_remote_video_url(monkeypatch, tmp_path: Path):
    output_dir = tmp_path / "output"
    url = "https://www.youtube.com/watch?v=example123"

    monkeypatch.setattr(tools, "OUTPUT_FOLDER", str(output_dir))
    monkeypatch.setattr(tools, "extract_any", lambda **_kwargs: "Video transcript")

    payload = json.loads(tools.extract_asset_text.func(url))

    assert payload["file_type"] == "video"
    assert payload["asset_path"] == url
    assert (
        Path(payload["output_path"]).read_text(encoding="utf-8") == "Video transcript"
    )


def test_detect_file_type_uses_url_suffixes():
    assert extractors.detect_file_type("https://example.com/test.pdf") == "pdf"
    assert extractors.detect_file_type("https://example.com/test.png") == "image"
    assert extractors.detect_file_type("https://example.com/test.mp3") == "audio"


def test_extract_asset_text_rejects_missing_files():
    try:
        tools.extract_asset_text.func("missing-file.png")
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError for a missing asset path")


def test_extract_printed_image_supports_remote_url(monkeypatch, tmp_path: Path):
    downloaded_image = tmp_path / "image.png"
    downloaded_image.write_bytes(b"fake-image")
    removed_dirs = []
    opened_paths = []

    monkeypatch.setattr(
        extractors,
        "resolve_remote_or_local_file",
        lambda _path, **_kwargs: (str(downloaded_image), str(tmp_path / "download")),
    )

    class StubImageModule:
        @staticmethod
        def open(path):
            opened_paths.append(path)
            return path

    class StubPytesseract:
        @staticmethod
        def image_to_string(image):
            return f"ocr:{image}"

    monkeypatch.setattr(
        extractors.shutil,
        "rmtree",
        lambda path, ignore_errors=True: removed_dirs.append(path),
    )

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pytesseract":
            return StubPytesseract
        if name == "PIL":
            return SimpleNamespace(Image=StubImageModule)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = extractors.extract_printed_image("https://example.com/test.png")

    assert result == f"ocr:{str(downloaded_image)}"
    assert opened_paths == [str(downloaded_image)]
    assert removed_dirs == [str(tmp_path / "download")]


def test_extract_pdf_supports_remote_url(monkeypatch, tmp_path: Path):
    downloaded_pdf = tmp_path / "doc.pdf"
    downloaded_pdf.write_bytes(b"pdf")
    removed_dirs = []

    monkeypatch.setattr(
        extractors,
        "resolve_remote_or_local_file",
        lambda _path, **_kwargs: (str(downloaded_pdf), str(tmp_path / "download")),
    )
    monkeypatch.setattr(
        extractors.shutil,
        "rmtree",
        lambda path, ignore_errors=True: removed_dirs.append(path),
    )

    class StubPage:
        def extract_text(self):
            return "Remote PDF text"

    class StubPdf:
        pages = [StubPage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pdfplumber":
            return SimpleNamespace(open=lambda _path: StubPdf())
        if name == "fitz":
            return SimpleNamespace(open=lambda _path: None)
        if name == "pytesseract":
            return SimpleNamespace(image_to_string=lambda _image: "")
        if name == "PIL":
            return SimpleNamespace(Image=SimpleNamespace(open=lambda _path: _path))
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = extractors.extract_pdf("https://example.com/file.pdf")

    assert "Remote PDF text" in result
    assert removed_dirs == [str(tmp_path / "download")]


def test_extract_excel_chunks_large_sheet(monkeypatch, tmp_path: Path):
    excel_path = tmp_path / "large.xlsx"
    excel_path.write_bytes(b"excel")

    class StubDataFrame:
        def __init__(self, rows):
            self.rows = rows
            self.empty = not rows

        def __len__(self):
            return len(self.rows)

        @property
        def iloc(self):
            class _ILoc:
                def __init__(self, outer):
                    self.outer = outer

                def __getitem__(self, item):
                    return StubDataFrame(self.outer.rows[item.start : item.stop])

            return _ILoc(self)

        def to_string(self, index=False):
            return "\n".join(self.rows)

    class StubPandas:
        @staticmethod
        def read_excel(_path, sheet_name=None):
            rows = [f"row {index}" for index in range(1, 451)]
            return {"Sheet1": StubDataFrame(rows)}

    monkeypatch.setattr(
        extractors,
        "resolve_remote_or_local_file",
        lambda _path, **_kwargs: (str(excel_path), None),
    )

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pandas":
            return StubPandas
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = extractors.extract_excel(str(excel_path))

    assert "--- ROWS 1 TO 200 OF 450 ---" in result
    assert "--- ROWS 201 TO 400 OF 450 ---" in result
    assert "--- ROWS 401 TO 450 OF 450 ---" in result


def test_extract_audio_supports_remote_url(monkeypatch, tmp_path: Path):
    downloaded_audio = tmp_path / "audio.mp3"
    downloaded_audio.write_bytes(b"audio")
    processed_audio = tmp_path / "processed.wav"
    processed_audio.write_bytes(b"processed-audio")
    removed_dirs = []
    transcribe_calls = []

    monkeypatch.setattr(
        extractors,
        "resolve_remote_or_local_file",
        lambda _path, **_kwargs: (str(downloaded_audio), str(tmp_path / "download")),
    )
    monkeypatch.setattr(
        extractors, "preprocess_audio", lambda _path: str(processed_audio)
    )
    monkeypatch.setattr(
        extractors,
        "get_audio_model",
        lambda: (
            "openai_whisper",
            SimpleNamespace(
                transcribe=lambda path, **_kwargs: transcribe_calls.append(path)
                or {"text": "remote audio"}
            ),
        ),
    )
    monkeypatch.setattr(extractors.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(extractors.os, "unlink", lambda _path: None)
    monkeypatch.setattr(
        extractors.shutil,
        "rmtree",
        lambda path, ignore_errors=True: removed_dirs.append(path),
    )

    transcript = extractors.extract_audio("https://example.com/audio.mp3")

    assert transcript == "remote audio"
    assert transcribe_calls == [str(processed_audio)]
    assert removed_dirs == [str(tmp_path / "download")]


def test_extract_audio_uses_faster_whisper_backend(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake-audio")
    processed_audio = tmp_path / "processed.wav"
    processed_audio.write_bytes(b"processed-audio")
    transcribe_calls = []

    class StubWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, _path, **_kwargs):
            transcribe_calls.append(_path)
            return (
                iter(
                    [
                        SimpleNamespace(text="hello"),
                        SimpleNamespace(text="world"),
                    ]
                ),
                {},
            )

    class StubFasterWhisperModule:
        WhisperModel = StubWhisperModel

    def fake_import_module(name: str):
        if name == "faster_whisper":
            return StubFasterWhisperModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(extractors, "_AUDIO_MODEL", None)
    monkeypatch.setattr(extractors, "_AUDIO_BACKEND", None)
    monkeypatch.setattr(extractors.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        extractors, "preprocess_audio", lambda _path: str(processed_audio)
    )
    monkeypatch.setattr(extractors.os, "unlink", lambda _path: None)

    transcript = extractors.extract_audio(str(audio_path))

    assert transcript == "hello world"
    assert transcribe_calls == [str(processed_audio)]


def test_extract_audio_retries_faster_whisper_when_first_attempt_is_empty(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake-audio")
    processed_audio = tmp_path / "processed.wav"
    processed_audio.write_bytes(b"processed-audio")
    call_options = []

    class StubWhisperModel:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def transcribe(self, _path, **kwargs):
            call_options.append(kwargs)
            self.calls += 1
            if self.calls == 1:
                return iter([SimpleNamespace(text="   ")]), {}
            return iter([SimpleNamespace(text="retry worked")]), {}

    class StubFasterWhisperModule:
        WhisperModel = StubWhisperModel

    def fake_import_module(name: str):
        if name == "faster_whisper":
            return StubFasterWhisperModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(extractors, "_AUDIO_MODEL", None)
    monkeypatch.setattr(extractors, "_AUDIO_BACKEND", None)
    monkeypatch.setattr(extractors.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        extractors, "preprocess_audio", lambda _path: str(processed_audio)
    )
    monkeypatch.setattr(extractors.os, "unlink", lambda _path: None)

    transcript = extractors.extract_audio(str(audio_path))

    assert transcript == "retry worked"
    assert len(call_options) == 2
    assert call_options[0]["vad_filter"] is False
    assert call_options[1]["temperature"] == 0.2


def test_extract_audio_falls_back_to_openai_whisper(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    processed_audio = tmp_path / "processed.wav"
    processed_audio.write_bytes(b"processed-audio")
    transcribe_calls = []

    class StubOpenAIWhisperModel:
        def transcribe(self, _path, **_kwargs):
            transcribe_calls.append(_path)
            return {"text": "fallback transcript"}

    class StubWhisperModule:
        @staticmethod
        def load_model(_name):
            return StubOpenAIWhisperModel()

    def fake_import_module(name: str):
        if name == "faster_whisper":
            raise ModuleNotFoundError(name)
        if name == "whisper":
            return StubWhisperModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(extractors, "_AUDIO_MODEL", None)
    monkeypatch.setattr(extractors, "_AUDIO_BACKEND", None)
    monkeypatch.setattr(extractors.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        extractors, "preprocess_audio", lambda _path: str(processed_audio)
    )
    monkeypatch.setattr(extractors.os, "unlink", lambda _path: None)

    transcript = extractors.extract_audio(str(audio_path))

    assert transcript == "fallback transcript"
    assert transcribe_calls == [str(processed_audio)]


def test_extract_audio_returns_message_when_transcript_is_empty(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    processed_audio = tmp_path / "processed.wav"
    processed_audio.write_bytes(b"processed-audio")

    class StubOpenAIWhisperModel:
        def transcribe(self, _path, **_kwargs):
            return {"text": "   "}

    class StubWhisperModule:
        @staticmethod
        def load_model(_name):
            return StubOpenAIWhisperModel()

    def fake_import_module(name: str):
        if name == "faster_whisper":
            raise ModuleNotFoundError(name)
        if name == "whisper":
            return StubWhisperModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(extractors, "_AUDIO_MODEL", None)
    monkeypatch.setattr(extractors, "_AUDIO_BACKEND", None)
    monkeypatch.setattr(extractors.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        extractors, "preprocess_audio", lambda _path: str(processed_audio)
    )
    monkeypatch.setattr(extractors.os, "unlink", lambda _path: None)

    transcript = extractors.extract_audio(str(audio_path))

    assert transcript.startswith("[No speech could be transcribed")


def test_extract_video_reuses_extract_audio(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")
    extracted_audio = tmp_path / "video-audio.wav"
    extracted_audio.write_bytes(b"audio")
    extracted_calls = []

    monkeypatch.setattr(
        extractors, "resolve_video_source", lambda _path: (str(video_path), None)
    )
    monkeypatch.setattr(
        extractors, "extract_audio_track_from_video", lambda _path: str(extracted_audio)
    )
    monkeypatch.setattr(
        extractors,
        "extract_audio",
        lambda _path: extracted_calls.append(_path) or "Video speech transcript",
    )
    monkeypatch.setattr(extractors.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(extractors.os, "unlink", lambda _path: None)

    transcript = extractors.extract_video(str(video_path))

    assert transcript == "Video speech transcript"
    assert extracted_calls == [str(extracted_audio)]


def test_extract_video_supports_remote_urls(monkeypatch, tmp_path: Path):
    remote_url = "https://youtu.be/example123"
    downloaded_video = tmp_path / "downloaded.mp4"
    downloaded_video.write_bytes(b"video")
    downloaded_dir = tmp_path / "download-dir"
    downloaded_dir.mkdir()
    extracted_audio = tmp_path / "remote-audio.wav"
    extracted_audio.write_bytes(b"audio")
    removed_dirs = []

    monkeypatch.setattr(
        extractors,
        "resolve_video_source",
        lambda _path: (str(downloaded_video), str(downloaded_dir)),
    )
    monkeypatch.setattr(
        extractors, "extract_audio_track_from_video", lambda _path: str(extracted_audio)
    )
    monkeypatch.setattr(extractors, "extract_audio", lambda _path: "Remote transcript")
    monkeypatch.setattr(extractors.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(extractors.os, "unlink", lambda _path: None)
    monkeypatch.setattr(
        extractors.shutil,
        "rmtree",
        lambda path, ignore_errors=True: removed_dirs.append(path),
    )

    transcript = extractors.extract_video(remote_url)

    assert transcript == "Remote transcript"
    assert removed_dirs == [str(downloaded_dir)]
