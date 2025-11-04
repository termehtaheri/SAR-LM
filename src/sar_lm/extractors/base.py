"""
ExtractorBase
-------------
Abstract base class for all audio feature extractors.
Ensures consistent interface and error handling across extractors.
"""

from __future__ import annotations
import abc
from typing import Any, Dict


class ExtractorBase(abc.ABC):
    """Abstract base for feature extractors."""

    name: str = "base"

    @abc.abstractmethod
    def run(self, audio_path: str) -> Dict[str, Any]:
        """Run feature extraction on a single audio file."""
        raise NotImplementedError

    @abc.abstractmethod
    def process_dir(self, input_dir: str, output_path: str) -> str:
        """Process a directory of audio files and save results."""
        raise NotImplementedError
