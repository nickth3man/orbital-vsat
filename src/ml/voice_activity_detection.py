"""
Voice Activity Detection (VAD) module for VSAT.

This module provides functionality for detecting speech segments in audio recordings.
It implements energy-based and ML-based approaches for VAD with configurable sensitivity.
"""

import logging
import numpy as np
import torch
import librosa
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Any
import os
import pickle
from enum import Enum

from .error_handling import ModelLoadError, InferenceError

logger = logging.getLogger(__name__)

class SensitivityPreset(Enum):
    """Enum representing sensitivity presets for voice activity detection.
    
    These presets control how sensitive the voice activity detector is to speech.
    Higher sensitivity will detect more potential speech segments, including quieter
    ones, while lower sensitivity will only detect clear, louder speech.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

    @classmethod
    def from_string(cls, value: str) -> 'SensitivityPreset':
        """Create a SensitivityPreset from a string value.
        
        Args:
            value: String representation of the preset
            
        Returns:
            SensitivityPreset: The corresponding preset
            
        Raises:
            ValueError: If the string does not match any preset
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_values = [preset.value for preset in cls]
            raise ValueError(
                f"Invalid sensitivity preset: '{value}'. "
                f"Valid options: {valid_values}"
            )

    def get_settings(self) -> Dict[str, Any]:
        """Get the VAD settings associated with this preset.
        
        Returns:
            Dict[str, Any]: Dictionary of VAD settings
        """
        settings = {
            self.HIGH.value: {
                "energy_threshold": 0.03,
                "min_speech_duration_ms": 200,
                "min_silence_duration_ms": 150,
                "speech_pad_ms": 50,
            },
            self.MEDIUM.value: {
                "energy_threshold": 0.05,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
                "speech_pad_ms": 30,
            },
            self.LOW.value: {
                "energy_threshold": 0.08,
                "min_speech_duration_ms": 300,
                "min_silence_duration_ms": 50,
                "speech_pad_ms": 20,
            },
            self.VERY_LOW.value: {
                "energy_threshold": 0.12,
                "min_speech_duration_ms": 400,
                "min_silence_duration_ms": 30,
                "speech_pad_ms": 10,
            }
        }
        return settings[self.value]

    def __str__(self) -> str:
        """Return a human-readable string representation of the preset."""
        return self.value.replace('_', ' ').title()

class VoiceActivityDetector:
    """Class for detecting speech segments in audio recordings."""
    
    DEFAULT_SETTINGS = {
        "energy_threshold": 0.05,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "window_size_ms": 30,
        "speech_pad_ms": 30,
        "smoothing_window_ms": 10,
        "use_model": True,
    }
    
    SENSITIVITY_PRESETS = {
        SensitivityPreset.HIGH: SensitivityPreset.HIGH.get_settings(),
        SensitivityPreset.MEDIUM: SensitivityPreset.MEDIUM.get_settings(),
        SensitivityPreset.LOW: SensitivityPreset.LOW.get_settings(),
        SensitivityPreset.VERY_LOW: SensitivityPreset.VERY_LOW.get_settings(),
    }
    
    def __init__(
        self, 
        auth_token: Optional[str] = None,
        device: str = "cpu",
        download_root: Optional[str] = None,
        settings: Optional[Dict] = None
    ):
        """Initialize the voice activity detector.
        
        Args:
            auth_token: HuggingFace authentication token
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
            settings: VAD settings to override defaults
        """
        self.device = device
        
        if download_root is None:
            download_root = str(Path.home() / '.vsat' / 'models' / 'vad')
        
        os.makedirs(download_root, exist_ok=True)
        
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        self.model = None
        if self.settings["use_model"]:
            try:
                from pyannote.audio import Pipeline
                
                self.model = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection", 
                    use_auth_token=auth_token
                )
                
                self.model.to(torch.device(device))
                
                logger.info(f"Loaded VAD model on {device}")
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                raise ModelLoadError(f"Failed to load VAD model: {e}")
        
        logger.info(
            f"Voice activity detector initialized (device: {device}, "
            f"use_model: {self.settings['use_model']})"
        )
    
    def apply_sensitivity_preset(self, preset: SensitivityPreset) -> None:
        """Apply a sensitivity preset to the VAD settings.
        
        Args:
            preset: Preset name (SensitivityPreset enum)
            
        Raises:
            ValueError: If preset name is invalid
        """
        if preset not in self.SENSITIVITY_PRESETS:
            valid_options = [preset.value for preset in self.SENSITIVITY_PRESETS.keys()]
            raise ValueError(
                f"Invalid sensitivity preset: {preset}. "
                f"Valid options: {valid_options}"
            )
        
        self.settings.update(self.SENSITIVITY_PRESETS[preset])
        logger.info(f"Applied {preset} sensitivity preset")
    
    def get_available_presets(self) -> List[SensitivityPreset]:
        """Get list of available sensitivity presets.
        
        Returns:
            List[SensitivityPreset]: List of preset names
        """
        return list(self.SENSITIVITY_PRESETS.keys())
    
    def detect_speech(
        self, 
        audio: Union[str, np.ndarray], 
        sample_rate: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict]:
        """Detect speech segments in audio.
        
        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate of the audio (required if audio is numpy array)
            progress_callback: Optional callback function for reporting progress (0-1)
            
        Returns:
            List[Dict]: List of speech segments with start/end times and confidence scores
            
        Raises:
            ValueError: If sample_rate is not provided for numpy array audio
            InferenceError: If an error occurs during inference
        """
        try:
            if isinstance(audio, str):
                audio_data, sample_rate = librosa.load(audio, sr=None)
            else:
                if sample_rate is None:
                    raise ValueError(
                        "Sample rate must be provided when audio is a numpy array"
                    )
                audio_data = audio
            
            if progress_callback:
                progress_callback(0.1)
            
            if self.model and self.settings["use_model"]:
                segments = self._detect_speech_ml(
                    audio_data, 
                    sample_rate, 
                    progress_callback
                )
            else:
                segments = self._detect_speech_energy(
                    audio_data, 
                    sample_rate, 
                    progress_callback
                )
            
            if progress_callback:
                progress_callback(1.0)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            raise InferenceError(f"Error detecting speech: {e}")
    
    def _detect_speech_ml(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict]:
        """Detect speech segments using ML model.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            progress_callback: Optional callback function for reporting progress
            
        Returns:
            List[Dict]: List of speech segments
        """
        waveform = torch.from_numpy(audio).float()
        waveform = waveform.unsqueeze(0)
        
        if progress_callback:
            progress_callback(0.3)
        
        vad_result = self.model({"waveform": waveform, "sample_rate": sample_rate})
        
        if progress_callback:
            progress_callback(0.8)
        
        segments = []
        for segment, track, score in vad_result.itertracks(yield_score=True):
            if track == "speech":
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": float(score)
                })
        
        segments = self._process_segments(segments)
        
        return segments
    
    def _detect_speech_energy(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict]:
        """Detect speech segments using energy-based approach.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            progress_callback: Optional callback function for reporting progress
            
        Returns:
            List[Dict]: List of speech segments
        """
        window_size = int(self.settings["window_size_ms"] * sample_rate / 1000)
        hop_length = window_size // 2
        
        if progress_callback:
            progress_callback(0.2)
        
        energy = np.array([
            sum(audio[i:i + window_size]**2) 
            for i in range(0, len(audio) - window_size, hop_length)
        ])
        energy = energy / np.max(energy)
        
        if progress_callback:
            progress_callback(0.4)
        
        smoothing_window = int(
            self.settings["smoothing_window_ms"] / 
            self.settings["window_size_ms"] * 2
        )
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            energy = np.convolve(energy, kernel, mode='same')
        
        if progress_callback:
            progress_callback(0.6)
        
        threshold = self.settings["energy_threshold"]
        speech_mask = energy > threshold
        
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_mask):
            if is_speech and not in_speech:
                in_speech = True
                start_frame = i
            elif not is_speech and in_speech:
                in_speech = False
                end_frame = i
                
                start_time = start_frame * hop_length / sample_rate
                end_time = end_frame * hop_length / sample_rate
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "confidence": float(np.mean(energy[start_frame:end_frame]))
                })
        
        if in_speech:
            end_frame = len(speech_mask)
            start_time = start_frame * hop_length / sample_rate
            end_time = end_frame * hop_length / sample_rate
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "confidence": float(np.mean(energy[start_frame:end_frame]))
            })
        
        if progress_callback:
            progress_callback(0.8)
        
        segments = self._process_segments(segments)
        
        return segments
    
    def _process_segments(self, segments: List[Dict]) -> List[Dict]:
        """Process speech segments to apply minimum duration and padding.
        
        Args:
            segments: List of speech segments
            
        Returns:
            List[Dict]: Processed speech segments
        """
        min_speech_duration = self.settings["min_speech_duration_ms"] / 1000
        min_silence_duration = self.settings["min_silence_duration_ms"] / 1000
        speech_pad = self.settings["speech_pad_ms"] / 1000
        
        segments = [
            segment for segment in segments 
            if segment["end"] - segment["start"] >= min_speech_duration
        ]
        
        for segment in segments:
            segment["start"] = max(0, segment["start"] - speech_pad)
            segment["end"] += speech_pad
        
        if segments:
            merged_segments = [segments[0]]
            
            for segment in segments[1:]:
                last_segment = merged_segments[-1]
                
                if segment["start"] - last_segment["end"] <= min_silence_duration:
                    last_segment["end"] = segment["end"]
                    last_segment["confidence"] = max(
                        last_segment["confidence"], 
                        segment["confidence"]
                    )
                else:
                    merged_segments.append(segment)
            
            return merged_segments
        
        return segments
    
    def save(self, path: str) -> None:
        """Save the VAD settings to a file.
        
        Args:
            path: Path to save the settings to
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.settings, f)
        
        logger.info(f"Saved VAD settings to {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'VoiceActivityDetector':
        """Load VAD settings from a file and create a new detector.
        
        Args:
            path: Path to load the settings from
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            VoiceActivityDetector: New detector with loaded settings
        """
        with open(path, 'rb') as f:
            settings = pickle.load(f)
        
        return cls(settings=settings, **kwargs)