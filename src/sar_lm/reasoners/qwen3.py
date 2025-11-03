# src/sar_lm/reasoners/qwen3.py
"""
Qwen3Reasoner
-------------
Production-ready wrapper for Qwen3 (30B) reasoning using symbolic features.
Currently supports the 'flat' mode (symbolic multimodal input).

Loads model from Hugging Face Hub and executes deterministic generation.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sar_lm.prompts.templates import REASONING_INSTRUCTION_PROMPT


class Qwen3Reasoner:
    """Qwen3 30B reasoning backend for symbolic audio reasoning tasks."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        log_dir: str = "logs",
    ):
        """Initialize tokenizer, model, and logger.

        Args:
            model_id (str): Hugging Face model ID.
            log_dir (str): Directory for run logs.
        """
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        ).eval()

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"run_qwen3_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode="w")],
        )
        self._log("✅ Qwen3Reasoner initialized successfully.")

    # ------------------------- public API -------------------------

    def predict(self, item: Dict[str, Any], merged_entry: Dict[str, Any]) -> Tuple[str, str]:
        """Run Qwen3 reasoning on a single item using flat symbolic features.

        Args:
            item (Dict[str, Any]): Question entry with 'question' and 'choices'.
            merged_entry (Dict[str, Any]): Symbolic features for the file.

        Returns:
            Tuple[str, str]: (final_answer, raw_decoded_output)
        """
        file_id = self._get_file_id(item)
        try:
            flat_prompt = self._generate_prompt_flat(merged_entry)
            content, raw = self._call_qwen_from_flat(flat_prompt, item["question"], item["choices"])
            self._log(f"[{file_id}] Answer: {content}")
            return content, raw
        except Exception as e:
            self._log(f"[ERROR] {file_id}: {e}")
            return f"[ERROR] {e}", ""

    def batch_predict(
        self,
        *,
        mmau_items: List[Dict[str, Any]],
        merged_features: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """Run Qwen3 reasoning for multiple MMAU items and save results.

        Args:
            mmau_items (List[Dict[str, Any]]): QA data.
            merged_features (List[Dict[str, Any]]): Symbolic features.
            output_path (Path): Path to write augmented JSON results.

        Returns:
            Path: Path to the saved JSON file.
        """
        feature_map = {e.get("file_id", e.get("id")): e for e in merged_features}

        if output_path.exists():
            augmented = self._load_json(output_path)
            processed_ids = {
                self._get_file_id(i) for i in augmented
            }
        else:
            augmented, processed_ids = [], set()

        for item in tqdm(mmau_items, desc="Running qwen3_flat"):
            file_id = self._get_file_id(item)
            if file_id in processed_ids:
                continue
            merged_entry = feature_map.get(file_id)
            if not merged_entry:
                self._log(f"[WARN] No symbolic features for {file_id}")
                continue

            final_answer, raw_decoded = self.predict(item, merged_entry)
            item["model_output"] = final_answer
            augmented.append(item)
            self._write_json(output_path, augmented)

        self._log(f"✅ Done. Results saved to {output_path}")
        return output_path

    # ------------------------- reasoning functions -------------------------

    @staticmethod
    def _generate_prompt_flat(entry: Dict[str, Any], music_thr: float = 0.5, speech_thr: float = 0.5) -> str:
        """Generate a multimodal symbolic reasoning prompt."""
        whisper = entry.get("whisper", {})
        panns = entry.get("panns", {})
        mt3 = entry.get("mt3", [])
        chordino = entry.get("chordino", [])
        musicnn = entry.get("musicnn", [])
        speech_emotion = entry.get("speech_emotion")

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

        lines = []
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
                    start, end = c.get("start"), c.get("end")
                    chord = c.get("chord")
                    if start is None or end is None:
                        lines.append(f"- chord={chord} (no timing info)")
                    else:
                        lines.append(f"- chord={chord} | start_sec={start:.3f} | end_sec={end:.3f}")
        else:
            lines.append("No significant musical content detected.")

        if has_speech and speech_emotion:
            emo = speech_emotion.get("speech_emotion") if isinstance(speech_emotion, dict) else speech_emotion
            if emo:
                lines.append(f"Detected speech emotion: {emo}.")
        elif not has_speech:
            lines.append("No significant speech detected.")

        return "\n".join(lines)

    def _build_llm_input(self, prompt: str, question: str, choices: List[str]) -> str:
        """Construct the reasoning instruction for Qwen using the shared template."""
        choices_str = "\n".join(choices)
        choices_examples = ", ".join(f'"{c}"' for c in choices)

        # use the shared reasoning prompt template
        instructions = REASONING_INSTRUCTION_PROMPT.format(choices_examples=choices_examples)

        return (
            "Audio Analysis Task\n\n"
            f"Question: {question}\n"
            f"Available choices (respond with EXACT text):\n"
            f"{choices_str}\n\n"
            f"Audio Context:\n{prompt}\n\n"
            f"Instructions:\n{instructions}\n\n"
            "Answer:"
        )

    def _call_qwen_from_flat(self, flat_prompt: str, question: str, choices: List[str]) -> Tuple[str, str]:
        """Call Qwen3 with the flat symbolic reasoning prompt."""
        llm_input = self._build_llm_input(flat_prompt, question, choices)
        messages = [
            {"role": "system", "content": "You are Qwen3, a model that answers multiple-choice questions about audio reasoning."},
            {"role": "user", "content": llm_input},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=512, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
        )

        full_decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        clean_answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return clean_answer, full_decoded

    # ------------------------- utils -------------------------

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
    def _get_file_id(item: Dict[str, Any]) -> str:
        return Path(item.get("audio_path", item.get("file_id", item.get("id")))).stem

    @staticmethod
    def _log(msg: str) -> None:
        logging.info(msg)
