# src/sar_lm/extractors/whisper.py
"""
WhisperExtractor
----------------
A symbolic feature extractor based on OpenAI's Whisper for automatic
speech transcription and language detection. It converts audio files
into structured, timestamped text segments suitable for reasoning and
caption-based pipelines.
"""

import os
import json
import torch
import whisper
import torchaudio
from tqdm import tqdm
from typing import Dict, Any, List
from .base import ExtractorBase


class WhisperExtractor(ExtractorBase):
    """Feature extractor that uses Whisper for transcribing and analyzing speech.

    This class loads audio, performs speech recognition with Whisper,
    and outputs structured symbolic features including language,
    full text, and timestamped segments.
    """

    name = "whisper"

    def __init__(self, model_name: str = "large", device: str | None = None):
        """Initialize the Whisper model.

        Args:
            model_name (str): Whisper model size (e.g., 'tiny', 'base', 'small', 'medium', 'large').
            device (str | None): Target device ('cpu' or 'cuda'). Automatically selected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=self.device)

    @staticmethod
    def load_audio(file_path: str, target_sr: int = 16000) -> torch.Tensor:
        """Load and resample an audio file.

        Args:
            file_path (str): Path to the audio file.
            target_sr (int): Target sample rate in Hz (default: 16000).

        Returns:
            torch.Tensor: Resampled mono waveform tensor.
        """
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0)

    def transcribe(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Transcribe an audio waveform into text and timestamped segments.

        Args:
            audio (torch.Tensor): Input waveform tensor.

        Returns:
            Dict[str, Any]: Transcription results containing:
                - 'language': Detected language.
                - 'full_text': Full transcription text.
                - 'segments': List of timestamped segment dictionaries.
        """
        audio_np = audio.numpy()
        result = self.model.transcribe(audio_np, language="en", fp16=(self.device == "cuda"))
        return {
            "language": result.get("language", "unknown"),
            "full_text": result.get("text", "").strip(),
            "segments": [
                {
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        }

    def run(self, audio_path: str) -> Dict[str, Any]:
        """Extract transcription features from a single audio file.

        Args:
            audio_path (str): Path to the input audio file.

        Returns:
            Dict[str, Any]: Symbolic transcription features including:
                - 'file_id': Audio file name without extension.
                - 'language': Detected language.
                - 'full_text': Full transcription text.
                - 'segments': Timestamped segment list.
        """
        audio = self.load_audio(audio_path)
        transcription = self.transcribe(audio)
        return {
            "file_id": os.path.splitext(os.path.basename(audio_path))[0],
            "language": transcription["language"],
            "full_text": transcription["full_text"],
            "segments": transcription["segments"],
        }

    def process_dir(self, input_dir: str, output_path: str) -> str:
        """Transcribe all audio files in a directory and save the results to JSON.

        Args:
            input_dir (str): Path to directory containing audio files (.wav, .mp3).
            output_path (str): Output JSON file path for saving results.

        Returns:
            str: Path to the saved JSON file.
        """
        results: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for fname in tqdm(os.listdir(input_dir), desc="Processing audio files with Whisper"):
            if fname.lower().endswith((".wav", ".mp3")):
                file_path = os.path.join(input_dir, fname)
                results.append(self.run(file_path))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path
