# src/sar_lm/pipelines/merge_features.py
"""
Merge Symbolic Audio Features
-----------------------------
Utility script to merge symbolic feature outputs from multiple extractors
(PANNs, Whisper, MT3, Musicnn, Chordino, and Emotion) into a unified JSON file.
Used as a post-processing step before reasoning or evaluation.
"""

import os
import json
from typing import Dict, Any, List


def merge_features(
    panns_path: str,
    whisper_path: str,
    mt3_path: str,
    emotion_path: str,
    musicnn_path: str,
    chordino_path: str,
    output_path: str,
) -> str:
    """Merge symbolic features from all extractors into one JSON file.

    Args:
        panns_path (str): Path to PANNs features JSON.
        whisper_path (str): Path to Whisper transcription JSON.
        mt3_path (str): Path to MT3 symbolic features JSON.
        emotion_path (str): Path to speech emotion JSON.
        musicnn_path (str): Path to Musicnn tag JSON.
        chordino_path (str): Path to Chordino chord progression JSON.
        output_path (str): Destination path for merged JSON file.

    Returns:
        str: Path to the saved merged JSON.
    """
    # === Load all JSONs ===
    with open(panns_path, "r") as f:
        panns_data = json.load(f)
    with open(whisper_path, "r") as f:
        whisper_data = json.load(f)
    with open(mt3_path, "r") as f:
        mt3_data = json.load(f)
    with open(emotion_path, "r") as f:
        emotion_data = json.load(f)
    with open(musicnn_path, "r") as f:
        musicnn_data = json.load(f)
    with open(chordino_path, "r") as f:
        chordino_data = json.load(f)

    # === Index non-list structures ===
    whisper_lookup = {item["file_id"]: item for item in whisper_data}
    emotion_lookup = {item["file_id"]: item for item in emotion_data}
    musicnn_lookup = {item["file_id"]: item for item in musicnn_data}

    # === Merge all features ===
    merged: List[Dict[str, Any]] = []
    for item in panns_data:
        file_id = item["file_id"]
        merged.append(
            {
                "file_id": file_id,
                "panns": {k: v for k, v in item.items() if k != "file_id"},
                "whisper": {
                    k: v for k, v in whisper_lookup.get(file_id, {}).items() if k != "file_id"
                },
                "mt3": mt3_data.get(file_id, []),
                "speech_emotion": emotion_lookup.get(file_id, {}).get("speech_emotion"),
                "musicnn": [tag for tag in musicnn_lookup.get(file_id, {}).get("tags", [])],
                "chordino": chordino_data.get(file_id, []),
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✓ Merged output saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example CLI usage (adjust paths accordingly)
    merge_features(
        panns_path="/data/.../panns_features.json",
        whisper_path="/data/.../whisper_large.json",
        mt3_path="/data/.../mt3_features.json",
        emotion_path="/data/.../speech_emotion_tags.json",
        musicnn_path="/data/.../musicnn_tags.json",
        chordino_path="/data/.../chord_progression.json",
        output_path="/data/.../merged_features.json",
    )
