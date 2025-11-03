# src/sar_lm/extractors/dawn_emotion.py
"""
DawnEmotionExtractor
--------------------
Symbolic speech emotion extractor using a fine-tuned Wav2Vec2 model
(`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`).
It predicts continuous Valence–Arousal–Dominance (VAD) scores and
converts them into discrete emotion categories (e.g., happy, calm, angry).
"""

import os
import json
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from typing import Dict, Any, List
from transformers import Wav2Vec2Processor
from .base import ExtractorBase


# === Emotion Model Definition ===
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torch.nn as nn


class RegressionHead(nn.Module):
    """Feed-forward regression head for continuous emotion prediction."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Wav2Vec2-based regression model for valence, arousal, dominance (VAD)."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0].mean(dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# === Main Extractor ===
class DawnEmotionExtractor(ExtractorBase):
    """Speech emotion extractor using Wav2Vec2-based regression (DAWN)."""

    name = "dawn_emotion"

    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"):
        """Initialize the Wav2Vec2 emotion model and processor.

        Args:
            model_name (str): Pretrained HuggingFace model ID.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Thresholds for VAD → emotion discretization (OmniBench default)
        self.VALENCE_THRESH = (0.45, 0.75)
        self.AROUSAL_THRESH = (0.45, 0.7)
        self.DOMINANCE_THRESH = (0.37, 0.6)

    def _process_func(self, x: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Predict emotion values (VAD) from raw audio signal.

        Args:
            x (np.ndarray): Input audio waveform.
            sr (int): Sampling rate of input signal.

        Returns:
            np.ndarray: Array of predicted [valence, arousal, dominance].
        """
        inputs = self.processor(x, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            _, logits = self.model(inputs)
        return logits.squeeze(0).cpu().numpy()

    def _bin_custom(self, value: float, low_th: float, high_th: float) -> str:
        """Discretize a continuous value into low/mid/high bins."""
        if value < low_th:
            return "low"
        elif value > high_th:
            return "high"
        return "mid"

    def _vad_to_emotion(self, v: float, a: float, d: float) -> str:
        """Map Valence–Arousal–Dominance values to a categorical emotion."""
        v_bin = self._bin_custom(v, *self.VALENCE_THRESH)
        a_bin = self._bin_custom(a, *self.AROUSAL_THRESH)
        d_bin = self._bin_custom(d, *self.DOMINANCE_THRESH)

        if v_bin == "high" and a_bin == "high" and d_bin == "high":
            return "happy"
        elif v_bin == "high" and a_bin == "low" and d_bin == "low":
            return "calm"
        elif v_bin == "low" and a_bin == "high" and d_bin == "high":
            return "angry"
        elif v_bin == "low" and a_bin == "high" and d_bin == "low":
            return "afraid"
        elif v_bin == "low" and a_bin == "low" and d_bin == "low":
            return "sad"
        elif v_bin == "mid" and a_bin == "high" and d_bin == "mid":
            return "surprised"
        elif v_bin == "mid" and a_bin == "low" and d_bin == "mid":
            return "bored"
        return "neutral"

    def run(self, audio_path: str) -> Dict[str, Any]:
        """Extract symbolic emotion features from a single audio file.

        Args:
            audio_path (str): Path to input audio file.

        Returns:
            Dict[str, Any]: Dictionary with file ID, continuous VAD scores, and discrete emotion tag.
        """
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            audio_np = waveform.squeeze(0).numpy()
            vad = self._process_func(audio_np, 16000)
            valence, arousal, dominance = map(float, vad)
            emotion = self._vad_to_emotion(valence, arousal, dominance)

            return {
                "file_id": file_id,
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "dominance": round(dominance, 4),
                "speech_emotion": emotion,
            }

        except Exception as e:
            return {"file_id": file_id, "error": str(e)}

    def process_dir(self, input_dir: str, output_path: str) -> str:
        """Process all audio files in a directory and save results as JSON.

        Args:
            input_dir (str): Directory with .wav/.mp3 files.
            output_path (str): Output JSON path for saving results.

        Returns:
            str: Path to the saved JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        results: List[Dict[str, Any]] = []
        for fname in tqdm(sorted(os.listdir(input_dir)), desc="Processing emotions"):
            if not fname.lower().endswith((".wav", ".mp3")):
                continue
            entry = self.run(os.path.join(input_dir, fname))
            results.append(entry)

            # Save after each file (incremental)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path
