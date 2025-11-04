"""
ExtractPipeline
---------------
Unified pipeline to run all symbolic feature extractors (Whisper, PANNs,
Musicnn, Chordino, MT3, DawnEmotion) on a directory of audio files.

Each extractor outputs its own JSON file under the specified output directory.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Type

from sar_lm.extractors.whisper import WhisperExtractor
# from sar_lm.extractors.panns import PANNsExtractor
# from sar_lm.extractors.musicnn import MusicnnExtractor
# from sar_lm.extractors.chordino import ChordinoExtractor
# from sar_lm.extractors.mt3 import MT3FeatureExtractor
# from sar_lm.extractors.dawn_emotion import DawnEmotionExtractor


# ------------------------------------------------------------------ #
# --------------------------- Logging ------------------------------ #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sar_lm.extract_pipeline")


# ------------------------------------------------------------------ #
# -------------------------- Pipeline ------------------------------ #
# ------------------------------------------------------------------ #

class ExtractPipeline:
    """High-level runner for all audio feature extractors."""

    def __init__(self, audio_dir: Path, output_dir: Path) -> None:
        """Initialize the extraction pipeline.

        Args:
            audio_dir: Directory containing audio clips (.wav/.mp3).
            output_dir: Directory where extracted JSON outputs are stored.
        """
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Register all extractors here
        self.extractors: List[Type] = [
            WhisperExtractor,
            # PANNsExtractor,
            # MusicnnExtractor,
            # ChordinoExtractor,
            # DawnEmotionExtractor,
            # MT3FeatureExtractor  # optional (requires heavy dependencies)
        ]

    def run(self, device: str | None = None) -> None:
        log.info(f"🎧 Starting feature extraction on: {self.audio_dir}")
        for extractor_cls in self.extractors:
            if extractor_cls.__name__ == "WhisperExtractor":
                extractor = extractor_cls(model_name="large", device=device)
            else:
                extractor = extractor_cls()
            output_path = self.output_dir / f"{extractor.name}_features.json"
            log.info(f"→ Running {extractor.name} extractor...")
            result_path = extractor.process_dir(str(self.audio_dir), str(output_path))
            log.info(f"✅ {extractor.name}: saved to {result_path}")
        log.info("🏁 All extractors completed successfully.")


# ------------------------------------------------------------------ #
# --------------------------- CLI Entry ---------------------------- #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAR-LM feature extraction pipeline.")
    parser.add_argument("--audio_dir", required=True, help="Path to input audio directory.")
    parser.add_argument("--output_dir", default="outputs/features", help="Directory to save feature files.")
    parser.add_argument("--include_mt3", action="store_true", help="Include MT3 transcription (slow).")
    parser.add_argument("--device", default=None, help="Force device (cpu or cuda).")
    args = parser.parse_args()

    pipeline = ExtractPipeline(Path(args.audio_dir), Path(args.output_dir))
    pipeline.run(device=args.device)


if __name__ == "__main__":
    main()
