# src/sar_lm/reasoners/qwen_omni.py
"""
QwenOmniReasoner
----------------
Production-ready wrapper for Qwen2.5-Omni reasoning using symbolic audio features.
Supports the 'flat' reasoning mode (text-only symbolic prompt).

Relies on the official Qwen2.5-Omni-7B model from Hugging Face.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class QwenOmniReasoner:
    """Qwen2.5-Omni (7B) symbolic reasoning engine."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-Omni-7B", log_dir: str = "logs"):
        """Initialize processor, model, and logging.

        Args:
            model_id (str): HF model identifier.
            log_dir (str): Directory for log files.
        """
        self.model_id = model_id
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"run_qwenomni_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_path, mode="w")],
        )
        self._log("✅ QwenOmniReasoner initialized.")

    # ---------------------- main API ----------------------

    def predict(self, item: Dict[str, Any], merged_entry: Dict[str, Any]) -> str:
        """Predict the answer for one item using flat symbolic reasoning.

        Args:
            item (Dict[str, Any]): Question object (must contain 'question', 'choices').
            merged_entry (Dict[str, Any]): Symbolic features for this file.

        Returns:
            str: Model output (verbatim choice).
        """
        file_id = self._file_id(item)
        try:
            flat_prompt = self._generate_prompt_flat(merged_entry)
            result = self._call_qwen_from_flat(flat_prompt, item["question"], item["choices"])
            self._log(f"[{file_id}] ✅ Answer: {result}")
            return result
        except Exception as e:
            err = f"[ERROR] {e}"
            self._log(err)
            return err

    def batch_predict(
        self,
        *,
        mmau_items: List[Dict[str, Any]],
        merged_features: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Run reasoning on all questions and save results.

        Args:
            mmau_items (List[Dict[str, Any]]): QA data.
            merged_features (List[Dict[str, Any]]): List of symbolic feature dicts.
            output_path (Path): JSON output path.

        Returns:
            Path: Path of saved augmented dataset.
        """
        fmap = {e.get("file_id", e.get("id")): e for e in merged_features}

        if output_path.exists():
            augmented = self._load_json(output_path)
            processed = {self._file_id(x) for x in augmented}
        else:
            augmented, processed = [], set()

        for item in tqdm(mmau_items, desc="Running qwen_omni_flat"):
            fid = self._file_id(item)
            if fid in processed:
                continue
            merged_entry = fmap.get(fid)
            if not merged_entry:
                self._log(f"[WARN] No features found for {fid}")
                continue

            answer = self.predict(item, merged_entry)
            item["model_output"] = answer
            augmented.append(item)
            self._write_json(output_path, augmented)

        self._log(f"✅ Done. Saved to {output_path}")
        return output_path

    # ---------------------- reasoning core ----------------------

    @staticmethod
    def _build_llm_input(prompt: str, question: str, choices: List[str]) -> str:
        """Assemble structured instruction prompt for Qwen-Omni."""
        choices_str = "\n".join(choices)
        choices_inline = ", ".join(f'"{c}"' for c in choices)
        instr = "\n".join([
            "You are a state-of-the-art reasoning model for audio understanding.",
            "Analyze the provided Audio Context to answer the Question accurately.",
            f"- Choose from: {choices_inline}",
            "Your answer must be exactly one of the options, no explanation."
        ])
        return (
            f"Audio Analysis Task\n\n"
            f"Question: {question}\nChoices:\n{choices_str}\n\n"
            f"Audio Context:\n{prompt}\n\n"
            f"Instructions:\n{instr}\n\nAnswer:"
        )

    def _call_qwen_from_flat(self, flat_prompt: str, question: str, choices: List[str]) -> str:
        """Generate answer from text prompt using Qwen-Omni."""
        llm_input = self._build_llm_input(flat_prompt, question, choices)
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen-Omni, an expert audio reasoning model."}]},
            {"role": "user", "content": [{"type": "text", "text": llm_input}]},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=128, return_audio=False)

        decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return decoded.split("assistant\n")[-1].strip()

    # ---------------------- prompt helpers ----------------------

    @staticmethod
    def _generate_prompt_flat(entry: Dict[str, Any], music_thr: float = 0.5, speech_thr: float = 0.5) -> str:
        """Generate a concise symbolic audio context prompt."""
        whisper, panns = entry.get("whisper", {}), entry.get("panns", {})
        mt3, chordino, musicnn = entry.get("mt3", []), entry.get("chordino", []), entry.get("musicnn", [])
        speech_em = entry.get("speech_emotion")

        def tag_present(panns: Dict[str, Any], keywords, thr=0.5):
            events = panns.get("clipwise_tags", []) + panns.get("timestamped_events", [])
            for e in events:
                if any(k in e.get("event", "").lower() for k in keywords) and e.get("confidence", 0) >= thr:
                    return True
            return False

        has_music = tag_present(panns, ["music", "singing", "instrument"], music_thr)
        has_speech = bool(whisper.get("full_text", "").strip()) or tag_present(
            panns, ["speech", "narration", "talk", "monologue"], speech_thr
        )

        lines: List[str] = []
        transcript = whisper.get("full_text", "").strip()
        lang = whisper.get("language", "unknown")
        lines.append(f"The audio is in {lang}.")
        if transcript:
            lines.append(f'Full transcript: "{transcript}".')
            for seg in whisper.get("segments", []):
                lines.append(f"- {seg['start']:.2f}s–{seg['end']:.2f}s: \"{seg['text'].strip()}\"")
        else:
            lines.append("No transcription text was detected.")

        if panns.get("duration"):
            lines.append(f"Audio duration: {panns['duration']:.2f} seconds.")

        if panns.get("clipwise_tags"):
            lines.append("Clipwise sound events:")
            for t in panns["clipwise_tags"]:
                lines.append(f"- event={t['event']} | confidence={t['confidence']:.3f}")

        if panns.get("timestamped_events"):
            lines.append("Timestamped sound events:")
            for ev in panns["timestamped_events"]:
                lines.append(
                    f"- event={ev['event']} | start_sec={ev['start']:.2f} | end_sec={ev['end']:.2f} "
                    f"| confidence={ev['confidence']:.3f}"
                )

        if has_music:
            if mt3:
                lines.append("Musical notes extracted from audio:")
                for n in mt3:
                    lines.append(
                        f"- instrument={n['instrument']} | note={n['note']} | pitch={n['pitch']} "
                        f"| start_sec={n['start']:.2f} | end_sec={n['end']:.2f}"
                    )
            if musicnn:
                lines.append("Music tags:")
                for t in musicnn:
                    lines.append(f"- tag={t['tag']} | confidence={t['score']:.3f}")
            if chordino:
                lines.append("Chord progression:")
                for c in chordino:
                    chord, start, end = c.get("chord"), c.get("start"), c.get("end")
                    if start is None or end is None:
                        lines.append(f"- chord={chord} (no timing info)")
                    else:
                        lines.append(f"- chord={chord} | start_sec={start:.3f} | end_sec={end:.3f}")
        else:
            lines.append("No significant musical content detected.")

        if has_speech and speech_em:
            emo = speech_em.get("speech_emotion") if isinstance(speech_em, dict) else speech_em
            if emo:
                lines.append(f"Detected speech emotion: {emo}.")
        elif not has_speech:
            lines.append("No significant speech detected.")

        return "\n".join(lines)

    # ---------------------- utils ----------------------

    @staticmethod
    def _file_id(item: Dict[str, Any]) -> str:
        return Path(item.get("audio_path", item.get("file_id", item.get("id")))).stem

    @staticmethod
    def _load_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _log(msg: str) -> None:
        logging.info(msg)
