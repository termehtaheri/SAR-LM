"""
MergeFeatures
-------------
Utility script to merge outputs from multiple SAR-LM feature extractors
(PANNs, Whisper, MT3, SpeechEmotion, Musicnn, Chordino) into a single
merged JSON file for reasoning.
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_features(
    panns_path: Path,
    whisper_path: Path,
    mt3_path: Path,
    emotion_path: Path,
    musicnn_path: Path,
    chordino_path: Path,
    output_path: Path,
) -> None:
    """Merge individual extractor outputs into a unified JSON list."""
    panns_data = load_json(panns_path)
    whisper_data = load_json(whisper_path)
    mt3_data = load_json(mt3_path)
    emotion_data = load_json(emotion_path)
    musicnn_data = load_json(musicnn_path)
    chordino_data = load_json(chordino_path)

    # === Convert to lookup dicts ===
    whisper_lookup = {x["file_id"]: x for x in whisper_data}
    emotion_lookup = {x["file_id"]: x for x in emotion_data}
    musicnn_lookup = {x["file_id"]: x for x in musicnn_data}
    mt3_lookup = {x["file_id"]: x["events"] for x in mt3_data}
    chordino_lookup = chordino_data  # already dict with file_ids as keys

    merged = []
    for item in panns_data:
        fid = item["file_id"]
        merged.append(
            {
                "file_id": fid,
                "panns": {k: v for k, v in item.items() if k != "file_id"},
                "whisper": {k: v for k, v in whisper_lookup.get(fid, {}).items() if k != "file_id"},
                "mt3": mt3_lookup.get(fid, []),
                "speech_emotion": emotion_lookup.get(fid, {}).get("speech_emotion"),
                "musicnn": list(musicnn_lookup.get(fid, {}).get("tags", [])),
                "chordino": chordino_lookup.get(fid, []),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged features saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge SAR-LM feature extractor outputs.")
    parser.add_argument("--panns", required=True)
    parser.add_argument("--whisper", required=True)
    parser.add_argument("--mt3", required=True)
    parser.add_argument("--emotion", required=True)
    parser.add_argument("--musicnn", required=True)
    parser.add_argument("--chordino", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merge_features(
        Path(args.panns),
        Path(args.whisper),
        Path(args.mt3),
        Path(args.emotion),
        Path(args.musicnn),
        Path(args.chordino),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
