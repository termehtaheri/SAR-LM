# src/sar_lm/extractors/chordino.py
"""
ChordinoExtractor
-----------------
Symbolic feature extractor for chord progression analysis using the
`chord_extractor` library (Chordino backend). It converts audio clips
into timestamped chord events for interpretable music reasoning.
"""

import os
import json
import soundfile as sf
import torchaudio
from typing import List, Dict, Any
from chord_extractor.extractors import Chordino
from .base import ExtractorBase


class ChordinoExtractor(ExtractorBase):
    """Chord progression extractor based on the Chordino backend."""

    name = "chordino"

    def __init__(self, target_sr: int = 44100):
        """Initialize the Chordino extractor.

        Args:
            target_sr (int): Target sample rate for preprocessing. Defaults to 44.1 kHz.
        """
        self.target_sr = target_sr
        self.model = Chordino()

    def _preprocess_audio(self, input_path: str, tmp_path: str) -> None:
        """Decode, convert to mono, resample, and write to a temporary WAV file.

        Args:
            input_path (str): Path to the source audio file (.wav or .mp3).
            tmp_path (str): Path to the temporary WAV file.
        """
        wav, sr = torchaudio.load(input_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        sf.write(tmp_path, wav.squeeze(0).numpy(), self.target_sr)

    def _extract_chords(self, wav_path: str) -> List[Dict[str, Any]]:
        """Run Chordino on a WAV file and extract timestamped chord events.

        Args:
            wav_path (str): Path to the preprocessed WAV file.

        Returns:
            List[Dict[str, Any]]: List of chord events with start/end timestamps.
        """
        seq = self.model.extract(wav_path)
        events = []

        for i, change in enumerate(seq):
            event = {
                "event": f"Chord: {change.chord}",
                "start": round(change.timestamp, 3),
                "end": round(seq[i + 1].timestamp, 3) if i + 1 < len(seq) else None,
            }
            events.append(event)

        return events

    def run(self, audio_path: str) -> Dict[str, Any]:
        """Extract chord progression events from a single audio file.

        Args:
            audio_path (str): Path to the input audio file.

        Returns:
            Dict[str, Any]: Symbolic features including file ID and chord sequence.
        """
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        tmp_path = "tmp_chordino.wav"

        try:
            self._preprocess_audio(audio_path, tmp_path)
            events = self._extract_chords(tmp_path)
            os.remove(tmp_path)

            return {"file_id": file_id, "chords": events}

        except Exception as e:
            # Cleanup temporary file on error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return {"file_id": file_id, "error": str(e)}

    def process_dir(self, input_dir: str, output_path: str) -> str:
        """Process all audio files in a directory and save chord sequences to JSON.

        Args:
            input_dir (str): Directory containing audio files.
            output_path (str): Destination JSON path.

        Returns:
            str: Path to the saved JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = {}

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith((".wav", ".mp3")):
                continue

            file_id = os.path.splitext(fname)[0]
            if file_id in all_data:
                continue  # Skip already processed

            entry = self.run(os.path.join(input_dir, fname))
            all_data[file_id] = entry.get("chords", entry.get("error", []))

            # Incremental save after each file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)

        return output_path
