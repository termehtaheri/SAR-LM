"""
GeminiMixedCaptioner
--------------------
Generates mixed (audio + symbolic) captions using Gemini 2.5 Pro.

Combines:
  • Raw audio input (.wav)
  • Symbolic feature summaries (Whisper, PANNs, MT3, etc.)
  • Agent-style feature selection (optional)

All text prompts are centralized in `sar_lm.prompts.templates`
for consistent reuse across captioning modes.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai

from sar_lm.prompts.templates import CAPTION_PROMPT_MIXED


class GeminiMixedCaptioner:
    """End-to-end + symbolic audio caption generator using Gemini 2.5 Pro."""

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: str | None = None) -> None:
        """Initialize Gemini client.

        Args:
            model_name: Gemini model name (default "gemini-2.5-pro").
            api_key: Optional API key; falls back to GEMINI_API_KEY.

        Raises:
            ValueError: If no API key is found.
        """
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    # ------------------------------------------------------------------ #
    # --------------------------- Helpers ------------------------------ #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tag_present(panns: Dict[str, Any], keywords: List[str], thr: float = 0.5) -> bool:
        events = panns.get("clipwise_tags", []) + panns.get("timestamped_events", [])
        for e in events:
            name = e.get("event", "").lower()
            conf = e.get("confidence", 0.0)
            if any(k in name for k in keywords) and conf >= thr:
                return True
        return False

    @staticmethod
    def _format_ts_events(events: List[Dict[str, Any]]) -> str:
        if not events:
            return "None"
        return "\n  - " + "\n  - ".join(
            f"{e['event']} from {e['start']:.2f}s to {e['end']:.2f}s" for e in events
        )

    @staticmethod
    def _format_whisper_segments(segments: List[Dict[str, Any]]) -> str:
        if not segments:
            return "None"
        return "\n  - " + "\n  - ".join(s["text"].strip() for s in segments if s.get("text"))

    # ------------------------------------------------------------------ #
    # --------------------------- Prompting ---------------------------- #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, entry: Dict[str, Any]) -> str:
        """Create a mixed-captioning feature summary (no agent filtering)."""
        whisper = entry.get("whisper", {})
        panns = entry.get("panns", {})
        mt3 = entry.get("mt3", [])
        musicnn = entry.get("musicnn", [])
        chordino = entry.get("chordino", [])
        speech_emotion = entry.get("speech_emotion")

        has_music = self._tag_present(panns, ["music", "singing", "instrument"])
        has_speech = bool(whisper.get("full_text", "").strip()) or self._tag_present(
            panns, ["speech", "narration", "talk", "monologue"]
        )

        duration = panns.get("duration")
        duration_str = f"{duration:.2f} s" if duration else "Unknown"
        clipwise_events = ", ".join(
            {t["event"] for t in panns.get("clipwise_tags", [])}
        ) or "None"
        timestamped_events = self._format_ts_events(panns.get("timestamped_events", []))

        # Musical features
        if has_music:
            lines = []
            if mt3:
                for n in mt3:
                    lines.append(
                        f"- {n.get('instrument')} plays {n.get('note')} "
                        f"(pitch={n.get('pitch')}) {n.get('start'):.2f}s–{n.get('end'):.2f}s"
                    )
            if musicnn:
                tags = [f"{t['tag']} ({t['score']:.2f})" for t in musicnn]
                lines.append("Musicnn tags: " + ", ".join(tags))
            if chordino:
                chords = [
                    f"- {c.get('chord')} {c.get('start',0):.2f}s–{c.get('end',0):.2f}s"
                    for c in chordino
                ]
                lines.append("Chord progression:\n  " + "\n  ".join(chords))
            music_features = "\n".join(lines)
        else:
            music_features = "No significant musical content detected."

        whisper_text = whisper.get("full_text", "").strip() or "None"
        whisper_segs = self._format_whisper_segments(whisper.get("segments", []))
        language_str = whisper.get("language", "Unknown")

        if has_speech and speech_emotion:
            emo = (
                speech_emotion.get("speech_emotion")
                if isinstance(speech_emotion, dict)
                else speech_emotion
            )
            emotion_str = f"Detected speech emotion: {emo}."
        else:
            emotion_str = "No significant speech detected."

        return (
            f"- Audio Duration: {duration_str}\n"
            f"- Detected Sound Events: {clipwise_events}\n"
            f"- Timestamped Sound Events: {timestamped_events}\n"
            f"- Musical Features:\n{music_features}\n"
            f"- Detected Language: {language_str}\n"
            f"- Speech Transcript: {whisper_text}\n"
            f"- Spoken Segments: {whisper_segs}\n"
            f"- {emotion_str}"
        )

    def _build_prompt_agent_style(self, entry: Dict[str, Any], selected_tools: List[str]) -> str:
        """Create a mixed feature summary using only selected tools."""
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
            lines.append(f"Language: {lang}.")
            if transcript:
                lines.append(f'Transcript: "{transcript}".')
            for seg in whisper.get("segments", []):
                lines.append(f"- {seg['start']:.2f}s–{seg['end']:.2f}s: \"{seg['text'].strip()}\"")

        if "PANNs" in selected_tools and panns:
            dur = panns.get("duration")
            if dur is not None:
                lines.append(f"Audio duration: {dur:.2f} s.")
            for t in panns.get("clipwise_tags", []):
                lines.append(f"- {t.get('event')} (conf={t.get('confidence',0):.2f})")

        if "MT3" in selected_tools and mt3:
            for n in mt3:
                lines.append(
                    f"- {n.get('instrument')} plays {n.get('note')} "
                    f"{n.get('start'):.2f}s–{n.get('end'):.2f}s"
                )

        if "Musicnn" in selected_tools and musicnn:
            for t in musicnn:
                lines.append(f"- {t['tag']} ({t['score']:.2f})")

        if "Chordino" in selected_tools and chordino:
            for c in chordino:
                start, end = c.get("start"), c.get("end")
                if start is not None and end is not None:
                    lines.append(f"- {c.get('chord')} {start:.2f}s–{end:.2f}s")
                else:
                    lines.append(f"- {c.get('chord')} (no timing)")

        if "speech_emotion" in selected_tools and speech_em:
            emo = (
                speech_em.get("speech_emotion")
                if isinstance(speech_em, dict)
                else speech_em
            )
            if emo:
                lines.append(f"Detected speech emotion: {emo}.")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # ------------------------- Captioning ----------------------------- #
    # ------------------------------------------------------------------ #

    def generate_caption(self, audio_path: Path, structured_info: str) -> str:
        """Generate caption combining symbolic text + raw audio."""
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        try:
            full_prompt = (
                f"--- INSTRUCTIONS ---\n{CAPTION_PROMPT_MIXED}\n\n"
                f"--- FEATURES ---\n{structured_info}"
            )
            response = self.model.generate_content(
                [
                    {
                        "role": "user",
                        "parts": [
                            {"text": full_prompt},
                            {"inline_data": {"mime_type": "audio/wav", "data": audio_bytes}},
                        ],
                    }
                ]
            )
            return (response.text or "").strip()
        except Exception as exc:  # pragma: no cover
            return f"[ERROR] {exc}"

    def batch_generate(
        self,
        audio_dir: Path,
        features_file: Path,
        output_file: Path,
        agent_file: Path | None = None,
        use_agent_style: bool = True,
        sleep_sec: float = 2.0,
    ) -> Path:
        """Generate mixed captions for all audio files in a directory.

        Args:
            audio_dir: Directory containing `.wav` clips.
            features_file: JSON of symbolic features.
            output_file: Path to save caption results.
            agent_file: Optional JSON of selected features (agent mode).
            use_agent_style: Whether to use agent feature filtering.
            sleep_sec: Delay between API calls (rate limiting).

        Returns:
            Path to the output JSON file.
        """
        with open(features_file, "r", encoding="utf-8") as f:
            symbolic_data = json.load(f)
        symbolic_map = {x["file_id"]: x for x in symbolic_data}

        selections: Dict[str, Any] = {}
        if agent_file and agent_file.exists():
            with open(agent_file, "r", encoding="utf-8") as f:
                selections = {e["id"]: e for e in json.load(f)}

        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
        else:
            results, done_ids = [], set()

        audio_files = sorted(p for p in os.listdir(audio_dir) if p.lower().endswith(".wav"))
        total = len(audio_files)
        print(f"📝 Starting mixed captioning: {total} files, {len(done_ids)} already done.")

        for fname in audio_files:
            file_id = Path(fname).stem
            if file_id in done_ids:
                continue

            audio_path = audio_dir / fname
            entry = symbolic_map.get(file_id)
            if entry is None:
                print(f"⚠️ No symbolic features for {file_id}; skipping.")
                continue

            if use_agent_style and file_id in selections:
                sel = selections[file_id].get("selected_features", [])
                structured_info = self._build_prompt_agent_style(entry, sel)
            else:
                structured_info = self._build_prompt(entry)

            caption = self.generate_caption(audio_path, structured_info)
            results.append({"id": file_id, "caption": caption})
            done_ids.add(file_id)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        print(f"🎉 Done. {len(results)} captions saved → {output_file}")
        return output_file
