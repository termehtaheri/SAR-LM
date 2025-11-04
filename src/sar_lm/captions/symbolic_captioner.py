"""
SymbolicCaptioner
-----------------
Generates symbolic audio captions using Gemini 2.5 Pro
from extracted symbolic features (Whisper, PANNs, MT3, etc.).

This module supports two modes:
  1. Standard symbolic captioning.
  2. Agent-style captioning (uses selected features from an external JSON).

All text prompts are stored in `sar_lm.prompts.templates`
to maintain a single source of truth.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai

from sar_lm.prompts.templates import CAPTION_PROMPT


class SymbolicCaptioner:
    """Gemini-based symbolic audio caption generator."""

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: str | None = None) -> None:
        """Initialize Gemini client.

        Args:
            model_name: Gemini model name.
            api_key: Optional explicit key; falls back to GEMINI_API_KEY.

        Raises:
            ValueError: If no API key is available.
        """
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    # ------------------------------------------------------------------ #
    # ------------------------ Prompt Builders ------------------------- #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tag_present(panns: Dict[str, Any], keywords: List[str], thr: float = 0.5) -> bool:
        """Check if any PANNs event matches given keywords above threshold."""
        events = panns.get("clipwise_tags", []) + panns.get("timestamped_events", [])
        for e in events:
            name = e.get("event", "").lower()
            conf = e.get("confidence", 0.0)
            if any(k in name for k in keywords) and conf >= thr:
                return True
        return False

    @staticmethod
    def _format_ts_events(events: List[Dict[str, Any]]) -> str:
        """Return human-readable timestamped events list."""
        if not events:
            return "None"
        return "\n  - " + "\n  - ".join(
            f"{e['event']} from {e['start']:.2f}s to {e['end']:.2f}s" for e in events
        )

    @staticmethod
    def _format_whisper_segments(segments: List[Dict[str, Any]]) -> str:
        """Format Whisper timestamped text segments."""
        if not segments:
            return "None"
        return "\n  - " + "\n  - ".join(
            s["text"].strip() for s in segments if s.get("text")
        )

    def _build_prompt(self, entry: Dict[str, Any]) -> str:
        """Assemble a symbolic captioning prompt from all features.

        Args:
            entry: Feature dictionary from merged_features.json.

        Returns:
            A fully formatted text prompt for Gemini captioning.
        """
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

        # ---------- Format all symbolic layers ----------
        duration = panns.get("duration")
        duration_str = f"{duration:.2f} seconds" if duration else "Unknown"
        clipwise_events = ", ".join(
            {t["event"] for t in panns.get("clipwise_tags", [])}
        ) or "None"
        timestamped_events = self._format_ts_events(panns.get("timestamped_events", []))

        # Music features
        if has_music:
            lines = []
            if mt3:
                lines.append("MT3 note events:")
                for n in mt3:
                    lines.append(
                        f"- {n.get('instrument')} plays {n.get('note')} (pitch={n.get('pitch')}) "
                        f"from {n.get('start'):.2f}s to {n.get('end'):.2f}s"
                    )
            if musicnn:
                tags = [f"{t['tag']} ({t['score']:.2f})" for t in musicnn]
                lines.append("Musicnn tags: " + ", ".join(tags))
            if chordino:
                chords = [
                    f"- {c.get('chord')} from {c.get('start', 0):.2f}s to {c.get('end', 0):.2f}s"
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

        symbolic_context = f"""- Audio Duration: {duration_str}
- Detected Sound Events: {clipwise_events}
- Timestamped Sound Events: {timestamped_events}
- Musical Features:
{music_features}
- Detected Language: {language_str}
- Speech Transcript (full): {whisper_text}
- Spoken Segments (timestamped): {whisper_segs}
- {emotion_str}"""

        return f"{CAPTION_PROMPT}\n\n--- FEATURES ---\n{symbolic_context}"

    def _build_prompt_agent_style(self, entry: Dict[str, Any], selected_tools: List[str]) -> str:
        """Assemble prompt including only selected tools (agent mode)."""
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
            dur = panns.get("duration")
            if dur is not None:
                lines.append(f"Audio duration: {dur:.2f} seconds.")
            clip = panns.get("clipwise_tags", [])
            for t in clip:
                lines.append(f"- event={t.get('event')} | confidence={t.get('confidence', 0):.3f}")
            ts = panns.get("timestamped_events", [])
            for ev in ts:
                lines.append(
                    f"- event={ev.get('event')} | start={ev.get('start', 0):.2f} | "
                    f"end={ev.get('end', 0):.2f} | conf={ev.get('confidence', 0):.3f}"
                )

        if "MT3" in selected_tools and mt3:
            for n in mt3:
                lines.append(
                    f"- instrument={n.get('instrument')} | note={n.get('note')} | "
                    f"pitch={n.get('pitch')} | start={n.get('start'):.2f} | end={n.get('end'):.2f}"
                )

        if "Musicnn" in selected_tools and musicnn:
            for t in musicnn:
                lines.append(f"- tag={t['tag']} | confidence={t['score']:.3f}")

        if "Chordino" in selected_tools and chordino:
            for c in chordino:
                start, end = c.get("start"), c.get("end")
                if start is None or end is None:
                    lines.append(f"- chord={c.get('chord')} (no timing info)")
                else:
                    lines.append(f"- chord={c.get('chord')} | start={start:.3f} | end={end:.3f}")

        if "speech_emotion" in selected_tools and speech_em:
            emo = (
                speech_em.get("speech_emotion")
                if isinstance(speech_em, dict)
                else speech_em
            )
            if emo:
                lines.append(f"Detected speech emotion: {emo}.")

        symbolic_context = "\n".join(lines)
        return f"{CAPTION_PROMPT}\n\n--- FEATURES ---\n{symbolic_context}"

    # ------------------------------------------------------------------ #
    # --------------------------- Generation --------------------------- #
    # ------------------------------------------------------------------ #

    def generate_caption(self, prompt: str) -> str:
        """Generate a symbolic caption for one prompt string."""
        try:
            response = self.model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as exc:  # pragma: no cover
            return f"[ERROR] {exc}"

    def batch_generate(
        self,
        feature_file: Path,
        output_file: Path,
        agent_file: Path | None = None,
        use_agent_style: bool = True,
        sleep_sec: float = 2.0,
    ) -> Path:
        """Generate symbolic captions for an entire dataset.

        Args:
            feature_file: Path to merged feature JSON file.
            output_file: Path to write captions JSON.
            agent_file: Optional path to JSON of selected features per file.
            use_agent_style: Whether to use agent-based prompts.
            sleep_sec: Delay between requests (rate limit safety).

        Returns:
            Path to the output JSON file.
        """
        with open(feature_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
        else:
            results, done_ids = [], set()

        selections: Dict[str, Any] = {}
        if agent_file and agent_file.exists():
            with open(agent_file, "r", encoding="utf-8") as f:
                selections = {e["id"]: e for e in json.load(f)}

        for entry in data:
            fid = entry["file_id"]
            if fid in done_ids:
                continue

            if use_agent_style and fid in selections:
                sel = selections[fid].get("selected_features", [])
                prompt = self._build_prompt_agent_style(entry, sel)
            else:
                prompt = self._build_prompt(entry)

            caption = self.generate_caption(prompt)
            results.append({"id": fid, "caption": caption})
            done_ids.add(fid)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        return output_file
