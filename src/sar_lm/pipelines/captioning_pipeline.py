"""
CaptioningPipeline
------------------
Unified interface for SAR-LM captioning modes:
  • symbolic  → Gemini symbolic captions (from extracted features)
  • mixed     → Gemini mixed captions (audio + features)
  • end2end   → Gemini audio-only captions

Example usage:
    PYTHONPATH=src python -m sar_lm.pipelines.captioning_pipeline \
        --mode symbolic \
        --audio_dir examples \
        --features outputs/features_merged/features_merged.json \
        --output outputs/captions/symbolic_captions.json
"""

from __future__ import annotations
import argparse
from pathlib import Path

from sar_lm.captions.symbolic_captioner import SymbolicCaptioner
from sar_lm.captions.mixed_captioner import GeminiMixedCaptioner
from sar_lm.captions.end2end_captioner import GeminiCaptioner

from dotenv import load_dotenv
load_dotenv()


class CaptioningPipeline:
    """Run Gemini captioners (symbolic, mixed, or end2end)."""

    def __init__(
        self,
        mode: str,
        audio_dir: Path,
        output_path: Path,
        features_path: Path | None = None,
        agent_path: Path | None = None,
    ) -> None:
        self.mode = mode.lower()
        self.audio_dir = audio_dir
        self.output_path = output_path
        self.features_path = features_path
        self.agent_path = agent_path

    def run(self) -> None:
        if self.mode == "symbolic":
            if not self.features_path:
                raise ValueError("--features is required for symbolic mode.")
            model = SymbolicCaptioner()
            model.batch_generate(
                feature_file=self.features_path,
                output_file=self.output_path,
                agent_file=self.agent_path,
                use_agent_style=False,
            )

        elif self.mode == "mixed":
            if not self.features_path:
                raise ValueError("--features is required for mixed mode.")
            model = GeminiMixedCaptioner()
            model.batch_generate(
                audio_dir=self.audio_dir,
                features_file=self.features_path,
                output_file=self.output_path,
                agent_file=self.agent_path,
                use_agent_style=False,
            )

        elif self.mode == "end2end":
            model = GeminiCaptioner()
            model.batch_generate(
                audio_dir=self.audio_dir,
                output_path=self.output_path,
            )

        else:
            raise ValueError(f"Unknown captioning mode: {self.mode}")

        print(f"✅ Done. Captions saved to {self.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAR-LM captioning pipeline.")
    parser.add_argument("--mode", required=True, choices=["symbolic", "mixed", "end2end"], help="Captioning mode")
    parser.add_argument("--audio_dir", required=True, help="Directory containing .wav audio clips")
    parser.add_argument("--features", help="Path to merged features JSON (required for symbolic/mixed)")
    parser.add_argument("--agent", help="Optional JSON with agent-selected tools")
    parser.add_argument("--output", default="outputs/captions/captions.json", help="Output JSON path")

    args = parser.parse_args()
    pipeline = CaptioningPipeline(
        mode=args.mode,
        audio_dir=Path(args.audio_dir),
        output_path=Path(args.output),
        features_path=Path(args.features) if args.features else None,
        agent_path=Path(args.agent) if args.agent else None,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
