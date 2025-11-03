# src/sar_lm/extractors/panns.py
"""
PANNsExtractor
---------------
A symbolic feature extractor using pretrained PANNs (Pretrained Audio Neural Networks)
for audio tagging and sound event detection. It produces both clip-level tags and
timestamped symbolic events suitable for interpretable reasoning tasks.
"""

import os
import json
import torch
import librosa
import numpy as np
import torchaudio
from tqdm import tqdm
from typing import List, Dict, Any
from panns_inference import AudioTagging, SoundEventDetection, labels
from .base import ExtractorBase


TARGET_SR = 32000


def load_audio(file_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load and resample an audio file to the target sample rate.

    This function first attempts to use Torchaudio for loading; if that fails,
    it falls back to Librosa for robust decoding of problematic MP3 headers.

    Args:
        file_path (str): Path to the input audio file.
        target_sr (int): Desired sample rate (default: 32 kHz).

    Returns:
        np.ndarray: Resampled mono waveform.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.squeeze(0).numpy()
    except Exception:
        y, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return y


def extract_clipwise_tags(
    audio: np.ndarray, model: AudioTagging, threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """Extract global clip-level tags using PANNs AudioTagging.

    Args:
        audio (np.ndarray): Input waveform array.
        model (AudioTagging): Pretrained audio tagging model.
        threshold (float): Minimum confidence threshold for keeping a tag.

    Returns:
        List[Dict[str, Any]]: List of detected events with confidence scores.
    """
    waveform_tensor = torch.tensor(audio)[None, :]
    clipwise_output, _ = model.inference(waveform_tensor)

    return [
        {"event": label, "confidence": float(score)}
        for label, score in zip(labels, clipwise_output[0])
        if score >= threshold
    ]


def extract_timestamped_events(
    framewise_output: np.ndarray, duration: float, threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """Convert framewise SED output into timestamped symbolic events.

    Args:
        framewise_output (np.ndarray): Framewise probabilities with shape (1, num_frames, num_classes).
        duration (float): Duration of the audio clip in seconds.
        threshold (float): Minimum activation threshold.

    Returns:
        List[Dict[str, Any]]: List of symbolic events with start/end times and confidence.
    """
    output_np = framewise_output[0]
    num_frames, num_classes = output_np.shape
    frame_hop = duration / num_frames
    events = []

    for class_idx in range(num_classes):
        active = output_np[:, class_idx] >= threshold
        if not np.any(active):
            continue

        label = labels[class_idx]
        indices = np.where(active)[0]

        starts, ends = [indices[0]], []
        for i in range(1, len(indices)):
            if indices[i] > indices[i - 1] + 1:
                ends.append(indices[i - 1])
                starts.append(indices[i])
        ends.append(indices[-1])

        for s_idx, e_idx in zip(starts, ends):
            start_time = round(s_idx * frame_hop, 2)
            end_time = round((e_idx + 1) * frame_hop, 2)
            confidence = float(output_np[s_idx : e_idx + 1, class_idx].max())
            events.append(
                {
                    "event": label,
                    "start": start_time,
                    "end": end_time,
                    "confidence": confidence,
                }
            )

    return events


class PANNsExtractor(ExtractorBase):
    """Symbolic feature extractor using PANNs models for tagging and event detection."""

    name = "panns"

    def __init__(self, device: str | None = None):
        """Initialize the PANNs models for audio tagging and sound event detection.

        Args:
            device (str | None): Target device ('cpu' or 'cuda'). Automatically selected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tagger = AudioTagging(checkpoint_path=None, device=self.device)
        self.detector = SoundEventDetection(checkpoint_path=None, device=self.device)

    def run(self, audio_path: str) -> Dict[str, Any]:
        """Extract symbolic audio features from a single file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Dict[str, Any]: Dictionary with file ID, duration, clipwise tags, and timestamped events.
        """
        audio = load_audio(audio_path)
        duration = len(audio) / TARGET_SR

        clipwise_tags = extract_clipwise_tags(audio, self.tagger)
        framewise_output = self.detector.inference(torch.tensor(audio)[None, :])
        timestamped_events = extract_timestamped_events(framewise_output, duration)

        return {
            "file_id": os.path.splitext(os.path.basename(audio_path))[0],
            "duration": round(duration, 2),
            "clipwise_tags": clipwise_tags,
            "timestamped_events": timestamped_events,
        }

    def process_dir(self, input_dir: str, output_path: str) -> str:
        """Process all audio files in a directory and save results to JSON.

        Args:
            input_dir (str): Path to directory containing audio files.
            output_path (str): Destination JSON file for saving extracted features.

        Returns:
            str: Path to the saved JSON file.
        """
        results: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for filename in tqdm(os.listdir(input_dir), desc="Processing audio files with PANNs"):
            if filename.lower().endswith((".wav", ".mp3")):
                file_path = os.path.join(input_dir, filename)
                results.append(self.run(file_path))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path
