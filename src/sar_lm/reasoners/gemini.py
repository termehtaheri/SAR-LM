# src/sar_lm/reasoners/gemini25.py
"""
GeminiReasoner
--------------
Production-safe Gemini 2.5 Pro reasoner supporting four modes:
  - 'gemini_direct'      : raw audio + instruction prompt
  - 'gemini_caption'     : caption text + instruction prompt
  - 'gemini_flat'        : flat symbolic features + instruction prompt
  - 'gemini_agent_audio' : agent-selected features + raw audio

API key is read from the environment variable GEMINI_API_KEY.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm
import google.generativeai as genai

from sar_lm.prompts.templates import REASONING_INSTRUCTION_PROMPT


class GeminiReasoner:
    """Unified Gemini 2.5 Pro reasoner with multiple prompting modes."""

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: Optional[str] = None, log_dir: str = "logs"):
        """Initialize the Gemini client and file logger.

        Args:
            model_name (str): Gemini model ID.
            api_key (Optional[str]): API key. If None, read from env GEMINI_API_KEY.
            log_dir (str): Directory to write run logs.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Export it before running.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_filename, mode="w")],
        )
        self._log("✅ GeminiReasoner initialized.")

        # Lazy, optional caches used by 'gemini_agent_audio'
        self._gemini_selections_map: Dict[str, Any] = {}
        self._symbolic_features_map: Dict[str, Any] = {}

    # ------------------------- public API -------------------------

    def predict(
        self,
        item: Dict[str, Any],
        mode: str,
        *,
        merged_entry: Optional[Dict[str, Any]] = None,
        audio_dir: Optional[Path] = None,
        caption: Optional[str] = None,
        agent_selected_tools: Optional[List[str]] = None,
    ) -> str:
        """Predict a single answer using the selected mode.

        Args:
            item (Dict[str, Any]): QA item with 'question' and 'choices'.
            mode (str): One of {'gemini_direct','gemini_caption','gemini_flat','gemini_agent_audio'}.
            merged_entry (Optional[Dict[str, Any]]): Merged symbolic features (for 'flat' or 'caption').
            audio_dir (Optional[Path]): Directory containing {file_id}.wav for audio modes.
            caption (Optional[str]): Caption string (for 'gemini_caption').
            agent_selected_tools (Optional[List[str]]): Tool list for agent mode.

        Returns:
            str: Model's final answer text.
        """
        file_id = self._item_file_id(item)
        question, choices = item["question"], item["choices"]

        try:
            if mode == "gemini_direct":
                if audio_dir is None:
                    return "[ERROR] audio_dir is required for gemini_direct."
                audio_path = audio_dir / f"{file_id}.wav"
                return self._call_gemini_direct(audio_path, question, choices)

            if mode == "gemini_caption":
                cap = caption or (merged_entry or {}).get("caption", "")
                return self._call_gemini_from_caption(cap, question, choices)

            if mode == "gemini_flat":
                if not merged_entry:
                    return "No flat features available."
                flat_prompt = self._generate_prompt_flat(merged_entry)
                return self._call_gemini_from_flat(flat_prompt, question, choices)

            if mode == "gemini_agent_audio":
                if audio_dir is None:
                    return "[ERROR] audio_dir is required for gemini_agent_audio."
                if not agent_selected_tools or not merged_entry:
                    return "[ERROR] Missing selected tools or symbolic features for agent mode."
                agent_prompt = self._generate_prompt_flat_agent_style(merged_entry, agent_selected_tools)
                audio_path = audio_dir / f"{file_id}.wav"
                return self._call_gemini_agent_audio(agent_prompt, audio_path, question, choices)

            return f"[ERROR] Unknown mode: {mode}"

        except Exception as e:
            return f"[ERROR] {e}"

    def batch_predict(
        self,
        *,
        mmau_items: List[Dict[str, Any]],
        output_path: Path,
        mode: str,
        merged_features: Optional[List[Dict[str, Any]]] = None,
        audio_dir: Optional[Path] = None,
        gemini_selections_path: Optional[Path] = None,
        symbolic_features_path: Optional[Path] = None,
    ) -> Path:
        """Run predictions in batch and save augmented items to JSON.

        Args:
            mmau_items (List[Dict[str, Any]]): QA items.
            output_path (Path): Output JSON path (augmented items).
            mode (str): Reasoning mode.
            merged_features (Optional[List[Dict[str, Any]]]): Merged symbolic features list (for 'flat'/'caption').
            audio_dir (Optional[Path]): Audio directory for audio modes.
            gemini_selections_path (Optional[Path]): JSON of agent selections (id→selected_features).
            symbolic_features_path (Optional[Path]): JSON of base symbolic features (id→entry).

        Returns:
            Path: Path to written JSON.
        """
        feature_map = self._index_by_id(merged_features) if merged_features else {}

        # optional caches for agent mode
        if gemini_selections_path:
            self._gemini_selections_map = self._safe_load_map(gemini_selections_path)
        if symbolic_features_path:
            self._symbolic_features_map = self._safe_load_map(symbolic_features_path, accept_list=True)

        if output_path.exists():
            augmented = self._load_json(output_path)
            processed_ids = {
                self._file_id_from_aug(item) for item in augmented
            }
        else:
            augmented, processed_ids = [], set()

        self._log(f"Processing {len(mmau_items)} entries with backend={mode}...")

        for item in tqdm(mmau_items, desc=f"Running {mode}"):
            file_id = self._item_file_id(item)
            if file_id in processed_ids:
                continue

            # prepare per-mode extras
            merged_entry = feature_map.get(file_id)
            caption = (merged_entry or {}).get("caption")
            selected_tools = None
            if mode == "gemini_agent_audio":
                sel_entry = self._gemini_selections_map.get(file_id, {})
                selected_tools = sel_entry.get("selected_features")
                merged_entry = merged_entry or self._symbolic_features_map.get(file_id)

            answer = self.predict(
                item,
                mode,
                merged_entry=merged_entry,
                audio_dir=audio_dir,
                caption=caption,
                agent_selected_tools=selected_tools,
            )

            self._log(f"FILE: {file_id}")
            self._log(f"QUESTION: {item['question']}")
            self._log(f"FINAL ANSWER: {answer}")
            self._log("-" * 80)

            item["model_output"] = answer
            augmented.append(item)
            self._write_json(output_path, augmented)

        self._log(f"✅ Done. Saved to {output_path}")
        return output_path

    # ------------------------- prompt builders -------------------------

    @staticmethod
    def _tag_present(panns: Dict[str, Any], keywords, thr: float = 0.5) -> bool:
        """Return True if any PANNs event matches a keyword with >= thr."""
        events = panns.get("clipwise_tags", []) + panns.get("timestamped_events", [])
        for e in events:
            name = str(e.get("event", "")).lower()
            conf = float(e.get("confidence", 0.0))
            if any(k in name for k in keywords) and conf >= thr:
                return True
        return False

    def _generate_prompt_flat(self, entry: Dict[str, Any], music_thr: float = 0.5, speech_thr: float = 0.5) -> str:
        """Build a flat prompt of symbolic features with modality-aware filtering."""
        whisper = entry.get("whisper", {})
        panns = entry.get("panns", {})

        mt3 = entry.get("mt3", [])
        chordino = entry.get("chordino", [])
        musicnn = entry.get("musicnn", [])
        speech_emotion = entry.get("speech_emotion")

        has_music = self._tag_present(panns, ["music", "singing", "instrument"], music_thr)
        has_speech = bool(whisper.get("full_text", "").strip()) or self._tag_present(
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

        duration = panns.get("duration")
        if duration is not None:
            lines.append(f"Audio duration: {duration:.2f} seconds.")

        clip = panns.get("clipwise_tags", [])
        if clip:
            lines.append("Clipwise sound events:")
            for t in clip:
                lines.append(f"- event={t.get('event')} | confidence={float(t.get('confidence', 0)):.3f}")

        ts_events = panns.get("timestamped_events", [])
        if ts_events:
            lines.append("Timestamped sound events:")
            for ev in ts_events:
                lines.append(
                    f"- event={ev.get('event')} | start_sec={float(ev.get('start', 0)):.2f} | "
                    f"end_sec={float(ev.get('end', 0)):.2f} | confidence={float(ev.get('confidence', 0)):.3f}"
                )

        if has_music:
            if mt3:
                lines.append("Musical notes extracted from audio:")
                for n in mt3:
                    lines.append(
                        f"- instrument={n.get('instrument')} | note={n.get('note')} | pitch={n.get('pitch')} "
                        f"| start_sec={float(n.get('start')):.2f} | end_sec={float(n.get('end')):.2f}"
                    )
            if musicnn:
                lines.append("Music tags:")
                for t in musicnn:
                    lines.append(f"- tag={t.get('tag')} | confidence={float(t.get('score', 0)):.3f}")
            if chordino:
                lines.append("Chord progression:")
                for c in chordino:
                    chord = c.get("chord")
                    start = c.get("start")
                    end = c.get("end")
                    if start is None or end is None:
                        lines.append(f"- chord={chord} (no timing info)")
                    else:
                        lines.append(f"- chord={chord} | start_sec={float(start):.3f} | end_sec={float(end):.3f}")
        else:
            lines.append("No significant musical content detected.")

        if has_speech and speech_emotion:
            emo = speech_emotion.get("speech_emotion") if isinstance(speech_emotion, dict) else speech_emotion
            if emo:
                lines.append(f"Detected speech emotion: {emo}.")
        elif not has_speech:
            lines.append("No significant speech detected.")

        return "\n".join(lines)

    @staticmethod
    def _generate_prompt_flat_agent_style(entry: Dict[str, Any], selected_tools: List[str]) -> str:
        """Build a flat prompt restricted by agent-selected tools."""
        lines: List[str] = []
        whisper = entry.get("whisper", {})
        panns = entry.get("panns", {})
        mt3 = entry.get("mt3", [])
        chordino = entry.get("chordino", [])
        musicnn = entry.get("musicnn", [])
        speech_em = entry.get("speech_emotion")

        if "Whisper-large" in selected_tools and whisper:
            transcript = whisper.get("full_text", "").strip()
            lang = whisper.get("language", "unknown")
            lines.append(f"The audio is in {lang}.")
            if transcript:
                lines.append(f'Full transcript: "{transcript}".')
                for seg in whisper.get("segments", []):
                    lines.append(f"- {seg['start']:.2f}s–{seg['end']:.2f}s: \"{seg['text'].strip()}\"")
            else:
                lines.append("No transcription text was detected.")

        if "PANNs" in selected_tools and panns:
            duration = panns.get("duration")
            if duration is not None:
                lines.append(f"Audio duration: {duration:.2f} seconds.")
            clip = panns.get("clipwise_tags", [])
            if clip:
                lines.append("Clipwise sound events:")
                for t in clip:
                    lines.append(f"- event={t.get('event')} | confidence={float(t.get('confidence', 0)):.3f}")
            ts_events = panns.get("timestamped_events", [])
            if ts_events:
                lines.append("Timestamped sound events:")
                for ev in ts_events:
                    lines.append(
                        f"- event={ev.get('event')} | start_sec={float(ev.get('start', 0)):.2f} | "
                        f"end_sec={float(ev.get('end', 0)):.2f} | confidence={float(ev.get('confidence', 0)):.3f}"
                    )

        if "MT3" in selected_tools and mt3:
            lines.append("Musical notes extracted from audio:")
            for n in mt3:
                lines.append(
                    f"- instrument={n.get('instrument')} | note={n.get('note')} | pitch={n.get('pitch')} "
                    f"| start_sec={float(n.get('start')):.2f} | end_sec={float(n.get('end')):.2f}"
                )

        if "Musicnn" in selected_tools and musicnn:
            lines.append("Music tags:")
            for t in musicnn:
                lines.append(f"- tag={t['tag']} | confidence={float(t.get('score', 0)):.3f}")

        if "Chordino" in selected_tools and chordino:
            lines.append("Chord progression:")
            for c in chordino:
                chord = c.get("chord")
                start = c.get("start")
                end = c.get("end")
                if start is None or end is None:
                    lines.append(f"- chord={chord} (no timing info)")
                else:
                    lines.append(f"- chord={chord} | start_sec={float(start):.3f} | end_sec={float(end):.3f}")

        if "speech_emotion" in selected_tools and speech_em:
            emo = speech_em.get("speech_emotion") if isinstance(speech_em, dict) else speech_em
            lines.append(f"Detected speech emotion: {emo}.") if emo else lines.append("No speech emotion detected.")

        return "\n".join(lines)

    @staticmethod
    def _build_llm_input(
        prompt: str,
        question: str,
        choices: List[str],
        header: str = "Audio Analysis Task",
    ) -> str:
        """Build the final LLM instruction string using the shared prompt template."""
        choices_str = "\n".join(choices)
        choices_examples = ", ".join(f'"{c}"' for c in choices)

        instructions = REASONING_INSTRUCTION_PROMPT.format(choices_examples=choices_examples)

        return (
            f"{header}\n\n"
            f"Question: {question}\n"
            f"Available choices (respond with EXACT text):\n"
            f"{choices_str}\n\n"
            f"Audio Context:\n"
            f"{prompt}\n\n"
            f"Instructions:\n"
            f"{instructions}\n\n"
            f"Answer:"
        )

    # ------------------------- gemini call wrappers -------------------------

    def _call_gemini_direct(self, audio_path: Path, question: str, choices: List[str]) -> str:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        llm_input = self._build_llm_input("(audio provided directly)", question, choices)
        resp = self.model.generate_content(
            [{"role": "user", "parts": [{"text": llm_input}, {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}}]}]
        )
        return (resp.text or "").strip()

    def _call_gemini_from_caption(self, caption: str, question: str, choices: List[str]) -> str:
        llm_input = self._build_llm_input(f"Audio caption: {caption}", question, choices)
        resp = self.model.generate_content([{"role": "user", "parts": [{"text": llm_input}]}])
        return (resp.text or "").strip()

    def _call_gemini_from_flat(self, flat_prompt: str, question: str, choices: List[str]) -> str:
        llm_input = self._build_llm_input(flat_prompt, question, choices)
        resp = self.model.generate_content([{"role": "user", "parts": [{"text": llm_input}]}])
        return (resp.text or "").strip()

    def _call_gemini_agent_audio(self, agent_prompt: str, audio_path: Path, question: str, choices: List[str]) -> str:
        llm_input = self._build_llm_input(agent_prompt, question, choices)
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        resp = self.model.generate_content(
            [{"role": "user", "parts": [{"text": llm_input}, {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}}]}]
        )
        return (resp.text or "").strip()

    # ------------------------- helpers -------------------------

    @staticmethod
    def _item_file_id(item: Dict[str, Any]) -> str:
        if "audio_path" in item:
            return Path(item["audio_path"]).stem
        return item.get("file_id", item.get("id"))

    @staticmethod
    def _file_id_from_aug(item: Dict[str, Any]) -> str:
        return Path(item["audio_path"]).stem if "audio_path" in item else item.get("file_id", item.get("id"))

    @staticmethod
    def _index_by_id(entries: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        if not entries:
            return {}
        out = {}
        for e in entries:
            k = e.get("file_id", e.get("id"))
            if k is not None:
                out[str(k)] = e
        return out

    @staticmethod
    def _load_json(path: Path) -> Any:
        if str(path).endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _safe_load_map(path: Path, *, accept_list: bool = False) -> Dict[str, Any]:
        try:
            data = GeminiReasoner._load_json(path)
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items()}
            if accept_list and isinstance(data, list):
                # list of dicts with 'file_id' or 'id'
                m: Dict[str, Any] = {}
                for e in data:
                    key = e.get("file_id", e.get("id"))
                    if key is not None:
                        m[str(key)] = e
                return m
        except Exception:
            pass
        return {}

    @staticmethod
    def _log(msg: str) -> None:
        logging.info(msg)
