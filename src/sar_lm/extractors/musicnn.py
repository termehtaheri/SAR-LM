# src/sar_lm/extractors/musicnn.py
"""
MusicnnExtractor
----------------
Symbolic feature extractor based on `musicnn` for high-level music tagging.
It predicts descriptive tags such as genre, instrumentation, or mood and returns
the top-N tags with confidence scores.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from musicnn.extractor import extractor
from .base import ExtractorBase


class MusicnnExtractor(ExtractorBase):
    """High-level music tagging extractor using pretrained `musicnn` models."""

    name = "musicnn"

    def __init__(self, model_name: str = "MTT_musicnn", top_k: int = 5):
        """Initialize the Musicnn extractor.

        Args:
            model_name (str): Model variant, usually 'MTT_musicnn' or 'MSD_musicnn'.
            top_k (int): Number of top tags to keep per file.
        """
        self.model_name = model_name
        self.top_k = top_k

    def run(self, audio_path: str) -> Dict[str, Any]:
        """Extract top-N music tags from a single audio file.

        Args:
            audio_path (str): Path to the input audio file (.wav or .mp3).

        Returns:
            Dict[str, Any]: Symbolic music features including:
                - 'file_id': Audio file name without extension.
                - 'tags': List of dictionaries {tag, score}.
                - 'error': Optional error message if extraction fails.
        """
        file_id = os.path.splitext(os.path.basename(audio_path))[0]

        try:
            taggram, tag_names = extractor(
                audio_path, model=self.model_name, extract_features=False
            )
            tag_likelihood = np.mean(taggram, axis=0)
            top_indices = tag_likelihood.argsort()[-self.top_k :][::-1]

            tags = [
                {"tag": tag_names[i], "score": float(tag_likelihood[i])}
                for i in top_indices
            ]

            return {"file_id": file_id, "tags": tags}

        except Exception as e:
            return {"file_id": file_id, "error": str(e)}

    def process_dir(self, input_dir: str, output_path: str) -> str:
        """Process all audio files in a directory and save tag results as JSON.

        Args:
            input_dir (str): Directory containing .wav or .mp3 files.
            output_path (str): Destination JSON file path.

        Returns:
            str: Path to the saved JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_ids = {entry["file_id"] for entry in results}
        else:
            results, processed_ids = [], set()

        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith((".wav", ".mp3")):
                continue

            file_id = os.path.splitext(fname)[0]
            if file_id in processed_ids:
                continue

            entry = self.run(os.path.join(input_dir, fname))
            results.append(entry)

            # Save progress after each file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path
