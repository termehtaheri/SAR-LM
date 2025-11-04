"""
GeminiCaptioner
---------------
Generates end-to-end audio captions using Gemini 2.5 Pro.

This module is import-only (no __main__). Prompts live in
`sar_lm.prompts.templates` to keep a single source of truth.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai

from sar_lm.prompts.templates import CAPTION_PROMPT_E2E


class GeminiCaptioner:
    """End-to-end audio caption generator using Gemini 2.5 Pro."""

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: str | None = None) -> None:
        """Initialize Gemini model client.

        Args:
            model_name: Gemini model identifier.
            api_key: Optional API key. If not provided, reads from GEMINI_API_KEY.
        """
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set. Export it before using GeminiCaptioner.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    def generate_caption(self, audio_path: Path) -> str:
        """Generate a concise caption for one audio file.

        Args:
            audio_path: Path to a .wav file.

        Returns:
            Caption string, or an error string prefixed with '[ERROR]'.
        """
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        try:
            response = self.model.generate_content(
                [
                    {
                        "role": "user",
                        "parts": [
                            {"text": CAPTION_PROMPT_E2E},
                            {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}},
                        ],
                    }
                ]
            )
            return (response.text or "").strip()
        except Exception as exc:  # pragma: no cover - upstream/network errors
            return f"[ERROR] {exc}"

    def batch_generate(self, audio_dir: Path, output_path: Path, sleep_sec: float = 2.0) -> Path:
        """Generate captions for all .wav files in a directory, with resumable writes.

        Args:
            audio_dir: Directory containing .wav files.
            output_path: JSON file to write [{"id": <file_id>, "caption": <str>}, ...].
            sleep_sec: Optional delay between requests to avoid rate limits.

        Returns:
            Path to the output JSON file.
        """
        audio_files: List[str] = sorted(
            f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")
        )

        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                results: List[Dict[str, str]] = json.load(f)
            done_ids = {row.get("id") for row in results}
        else:
            results, done_ids = [], set()

        for fname in audio_files:
            file_id = Path(fname).stem
            if file_id in done_ids:
                continue

            caption = self.generate_caption(audio_dir / fname)
            results.append({"id": file_id, "caption": caption})
            done_ids.add(file_id)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        return output_path
