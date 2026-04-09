from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def Listen(prompt: Optional[str] = None) -> str:
    """
    Listen to the user's voice from the microphone and return transcribed text.

    Implementation:
    - Offline STT via Vosk (local model) + PyAudio microphone capture.
    - If Vosk (or model) isn't available, falls back to typed input.

    Setup:
    - Download a Vosk model (example: `vosk-model-small-en-us-0.15`)
    - Put it in the project root (same folder as `main.py`), OR set env var `VOSK_MODEL_PATH`
    """
    if prompt:
        print(prompt)

    try:
        import pyaudio  # type: ignore
        from vosk import KaldiRecognizer, Model  # type: ignore
    except Exception:
        return input("Input:")

    model_dir = os.environ.get("VOSK_MODEL_PATH")
    if model_dir:
        model_path = Path(model_dir)
    else:
        model_path = Path.cwd() / "vosk-model-small-en-us-0.15"

    if not model_path.exists():
        print(
            "Offline STT model not found. Download a Vosk model and place it at "
            f"'{model_path}'. Falling back to typed input."
        )
        return input("Input:")

    stream = None
    pa = None
    try:
        model = Model(str(model_path))
        recognizer = KaldiRecognizer(model, 16000)

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000,
        )
        stream.start_stream()

        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result() or "{}")
                text = (result.get("text") or "").strip()
                if text:
                    return text
                return input("Input:")
    except Exception:
        return input("Input:")
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass