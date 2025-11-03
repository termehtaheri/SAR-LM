# src/sar_lm/agents/selector.py
"""
SAR-LM | Gemini Agentic Feature Selector
----------------------------------------

This module uses Gemini 2.5 Pro to automatically decide which symbolic features
(Whisper, PANNs, MT3, Musicnn, Chordino, Emotion) are most relevant for
audio reasoning tasks.

For each audio file, it listens to the audio and reviews the full symbolic
feature bundle, then outputs a JSON object with:
    - selected_features: list[str]
    - use_caption: bool
    - comment: str

Example usage:
    from sar_lm.agents.selector import GeminiFeatureSelector
    selector = GeminiFeatureSelector(api_key="YOUR_KEY")
    selector.run(
        audio_dir="../dataset/mmau/test-mini-audios",
        features_file="../dataset/features/mmau/merged_features.json",
        output_path="../dataset/features/mmau/gemini_selected_features_full.json"
    )
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import google.generativeai as genai
from sar_lm.prompts.templates import FEATURE_SELECTOR_PROMPT


class GeminiFeatureSelector:
    """Agentic feature selector using Gemini 2.5 Pro."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro",
        log_dir: str = "logs",
        prompt_text: str = FEATURE_SELECTOR_PROMPT,
    ):
        """
        Initialize GeminiFeatureSelector.

        Args:
            api_key: Google API key.
            model_name: Gemini model name (default: "gemini-2.5-pro").
            log_dir: Directory for logs.
            prompt_text: Instruction text for Gemini.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.prompt_text = prompt_text

        os.makedirs(log_dir, exist_ok=True)
        log_file = Path(log_dir) / "feature_selector.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode="a")],
        )
        logging.info("✅ GeminiFeatureSelector initialized.")

    # -------------------------------------------------------------------------
    def _process_audio(
        self,
        audio_path: Path,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Gemini on a single audio file + symbolic feature bundle."""
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            feature_blob = json.dumps({
                "whisper": features.get("whisper"),
                "panns": features.get("panns"),
                "mt3": features.get("mt3"),
                "chordino": features.get("chordino"),
                "musicnn": features.get("musicnn"),
                "speech_emotion": features.get("speech_emotion"),
            }, indent=2)

            response = self.model.generate_content([
                {
                    "role": "user",
                    "parts": [
                        {"text": self.prompt_text},
                        {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}},
                        {"text": f"Full symbolic features:\n{feature_blob}"}
                    ],
                }
            ])

            json_text = response.text.strip()
            json_text = re.sub(r"^```(?:json)?|```$", "", json_text, flags=re.DOTALL).strip()

            try:
                return json.loads(json_text)
            except Exception as err:
                logging.error(f"JSON parse error ({audio_path.name}): {err}")
                return {
                    "selected_features": [],
                    "use_caption": False,
                    "comment": f"[ERROR parsing JSON] Raw output: {json_text[:300]}"
                }

        except Exception as e:
            logging.error(f"Gemini API error ({audio_path.name}): {e}")
            time.sleep(60)
            return {
                "selected_features": [],
                "use_caption": False,
                "comment": f"[ERROR] {e}"
            }

    # -------------------------------------------------------------------------
    def run(
        self,
        *,
        audio_dir: str | Path,
        features_file: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Run feature selection for all audio files in directory.

        Args:
            audio_dir: Directory containing audio clips (.wav).
            features_file: Path to merged symbolic features JSON.
            output_path: Path to save Gemini-selected features.

        Returns:
            Path to saved output JSON.
        """
        audio_dir = Path(audio_dir)
        features_file = Path(features_file)
        output_path = Path(output_path)

        with open(features_file, "r") as f:
            symbolic_data: List[Dict[str, Any]] = json.load(f)
        symbolic_map = {entry["file_id"]: entry for entry in symbolic_data}

        if output_path.exists():
            with open(output_path, "r") as f:
                results = json.load(f)
            done_ids = {x["id"] for x in results}
        else:
            results, done_ids = [], set()

        audio_files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))
        total = len(audio_files)
        print(f"🧠 Starting feature selection: {total} total, {len(done_ids)} already done.")

        for i, fname in enumerate(audio_files, 1):
            file_id = Path(fname).stem
            if file_id in done_ids:
                continue

            entry = symbolic_map.get(file_id)
            if entry is None:
                logging.warning(f"No features found for {file_id}, skipping.")
                continue

            result = self._process_audio(audio_dir / fname, entry)
            result["id"] = file_id
            results.append(result)
            done_ids.add(file_id)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ [{len(results)}/{total}] {file_id}: "
                  f"{result.get('selected_features', [])} | caption={result.get('use_caption')}")

        print(f"\n🎯 Done! Saved to {output_path}")
        return output_path
