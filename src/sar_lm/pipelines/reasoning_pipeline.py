# src/sar_lm/pipelines/reasoning_pipeline.py
"""
ReasoningPipeline
-----------------
Unified pipeline for running SAR-LM reasoners (Gemini, Qwen3, Qwen-Omni).
It loads merged symbolic features and question–answer items (MMAU-style),
then dispatches reasoning tasks to the selected backend.

Example:
    PYTHONPATH=src python -m sar_lm.pipelines.reasoning_pipeline \
        --reasoner qwen3 \
        --features outputs/features_merged.json \
        --qa examples/sample_qa.json \
        --output outputs/qwen3_results.json
"""


from __future__ import annotations
import argparse
import json
from pathlib import Path
from sar_lm.reasoners.gemini import GeminiReasoner
from sar_lm.reasoners.qwen3 import Qwen3Reasoner
from sar_lm.reasoners.qwen_omni import QwenOmniReasoner


class ReasoningPipeline:
    """Runs a selected reasoner on symbolic features and QA data."""
    def __init__(self, reasoner: str, features_path: Path, qa_path: Path, output_path: Path):
        self.reasoner = reasoner.lower()
        self.features_path = features_path
        self.qa_path = qa_path
        self.output_path = output_path

    def run(self) -> None:
        with open(self.qa_path, "r", encoding="utf-8") as f:
            mmau_items = json.load(f)

        with open(self.features_path, "r", encoding="utf-8") as f:
            merged_features = json.load(f)

        if self.reasoner == "gemini":
            model = GeminiReasoner()
            model.batch_predict(
                mmau_items=mmau_items,
                merged_features=merged_features,
                output_path=self.output_path,
                mode="gemini_flat",
            )

        elif self.reasoner == "qwen3":
            model = Qwen3Reasoner()
            model.batch_predict(
                mmau_items=mmau_items,
                merged_features=merged_features,
                output_path=self.output_path,
            )

        elif self.reasoner == "qwen_omni":
            model = QwenOmniReasoner()
            model.batch_predict(
                mmau_items=mmau_items,
                merged_features=merged_features,
                output_path=self.output_path,
            )

        else:
            raise ValueError(f"Unknown reasoner: {self.reasoner}")

        print(f"✅ Done. Results saved to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SAR-LM reasoning pipeline.")
    parser.add_argument("--reasoner", required=True, choices=["gemini", "qwen3", "qwen_omni"], help="Select reasoning backend")
    parser.add_argument("--features", required=True, help="Path to merged symbolic features JSON")
    parser.add_argument("--qa", required=True, help="Path to QA JSON file")
    parser.add_argument("--output", default="outputs/reasoning_results.json", help="Output JSON path")
    args = parser.parse_args()

    pipeline = ReasoningPipeline(
        args.reasoner,
        Path(args.features),
        Path(args.qa),
        Path(args.output),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
