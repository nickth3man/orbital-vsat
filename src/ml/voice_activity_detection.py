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
from typing import List, Tuple, Dict, Optional, Union, Callable
import os
import pickle

from .error_handling import ModelLoadError, InferenceError

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """Class for detecting speech segments in audio recordings."""
    
    # Default VAD settings
    DEFAULT_SETTINGS = {
        "energy_threshold": 0.05,  # Energy threshold (0.0-1.0)
        "min_speech_duration_ms": 250,  # Minimum speech segment duration in ms
        "min_silence_duration_ms": 100,  # Minimum silence duration to split segments in ms
        "window_size_ms": 30,  # Analysis window size in ms
        "speech_pad_ms": 30,  # Padding to add around speech segments in ms
        "smoothing_window_ms": 10,  # Window for smoothing the energy curve in ms
        "use_model": True,  # Whether to use ML model or energy-based approach
    }
    
    # Sensitivity presets
    SENSITIVITY_PRESETS = {
        "high": {  # More sensitive, detects more speech
            "energy_threshold": 0.03,
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 50,
        },
        "medium": {  # Balanced approach
            "energy_threshold": 0.05,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 30,
        },
        "low": {  # Less sensitive, only detects clear speech
            "energy_threshold": 0.08,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 50,
            "speech_pad_ms": 20,
        },
        "very_low": {  # Minimal sensitivity, only detects loud speech
            "energy_threshold": 0.12,
            "min_speech_duration_ms": 400,
            "min_silence_duration_ms": 30,
            "speech_pad_ms": 10,
        }
    }
    
    def __init__(self, 
                 auth_token: Optional[str] = None,
                 device: str = "cpu",
                 download_root: Optional[str] = None,
                 settings: Optional[Dict] = None):
        """Initialize the voice activity detector.
        
        Args:
            auth_token: HuggingFace authentication token
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
            settings: VAD settings to override defaults
        """
        self.device = device
        
        # Set default download root if not provided
        if download_root is None:
            download_root = str(Path.home() / '.vsat' / 'models' / 'vad')
        
        # Create download directory if it doesn't exist
        os.makedirs(download_root, exist_ok=True)
        
        # Initialize settings with defaults
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Initialize ML model if enabled
        self.model = None
        if self.settings["use_model"]:
            try:
                # Import here to avoid dependency if not using ML model
                from pyannote.audio import Pipeline
                
                # Initialize the VAD model
                self.model = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection", 
                    use_auth_token=auth_token
                )
                
                # Move model to specified device
                self.model.to(torch.device(device))
                
                logger.info(f"Loaded VAD model on {device}")
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                raise ModelLoadError(f"Failed to load VAD model: {e}")
        
        logger.info(f"Voice activity detector initialized (device: {device}, use_model: {self.settings['use_model']})")
    
    def apply_sensitivity_preset(self, preset: str) -> None:
        """Apply a sensitivity preset to the VAD settings.
        
        Args:
            preset: Preset name ("high", "medium", "low", "very_low")
            
        Raises:
            ValueError: If preset name is invalid
        """
        if preset not in self.SENSITIVITY_PRESETS:
            raise ValueError(f"Invalid sensitivity preset: {preset}. Valid options: {list(self.SENSITIVITY_PRESETS.keys())}")
        
        # Update settings with preset values
        self.settings.update(self.SENSITIVITY_PRESETS[preset])
        logger.info(f"Applied {preset} sensitivity preset")
    
    def get_available_presets(self) -> List[str]:
        """Get list of available sensitivity presets.
        
        Returns:
            List[str]: List of preset names
        """
        return list(self.SENSITIVITY_PRESETS.keys())
    
    def detect_speech(self, 
                     audio: Union[str, np.ndarray], 
                     sample_rate: Optional[int] = None,
                     progress_callback: Optional[Callable[[float], None]] = None) -> List[Dict]:
        """Detect speech segments in audio.
        
        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate of the audio (required if audio is numpy array)
            progress_callback: Optional callback function for reporting progress (0-1)
            
        Returns:
            List[Dict]: List of speech segments with start/end times and confidence scores
            
        Raises:
            ValueError: If sample_rate is not provided for numpy array input
            InferenceError: If speech detection fails
        """
        try:
            # Handle different input types
            if isinstance(audio, str):
                # Audio is a file path
                audio_data, sample_rate = librosa.load(audio, sr=None, mono=True)
            elif isinstance(audio, np.ndarray):
                # Audio is a numpy array
                if sample_rate is None:
                    raise ValueError("Sample rate must be provided when audio is a numpy array")
                audio_data = audio
                # Convert to mono if stereo
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
            else:
                raise TypeError(f"Unsupported audio type: {type(audio)}")
            
            # Use ML model if enabled
            if self.settings["use_model"] and self.model:
                return self._detect_speech_ml(audio_data, sample_rate, progress_callback)
            else:
                return self._detect_speech_energy(audio_data, sample_rate, progress_callback)
                
        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            raise InferenceError(f"Speech detection failed: {e}")
    
    def _detect_speech_ml(self, 
                         audio_data: np.ndarray, 
                         sample_rate: int,
                         progress_callback: Optional[Callable[[float], None]] = None) -> List[Dict]:
        """Detect speech segments using ML model.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            progress_callback: Optional callback function for reporting progress (0-1)
            
        Returns:
            List[Dict]: List of speech segments with start/end times and confidence scores
        """
        try:
            # Convert to waveform expected by pyannote
            waveform = torch.tensor(audio_data).unsqueeze(0)
            
            # Apply VAD
            vad_result = self.model({"waveform": waveform, "sample_rate": sample_rate})
            
            # Extract speech segments
            segments = []
            for speech_region in vad_result.get_timeline().support():
                segment = {
                    "start": speech_region.start,
                    "end": speech_region.end,
                    "duration": speech_region.end - speech_region.start,
                    "confidence": 0.95  # Default confidence for ML model
                }
                segments.append(segment)
            
            # Apply minimum duration filtering
            min_duration_sec = self.settings["min_speech_duration_ms"] / 1000
            segments = [s for s in segments if s["duration"] >= min_duration_sec]
            
            # Apply speech padding
            pad_sec = self.settings["speech_pad_ms"] / 1000
            for segment in segments:
                segment["start"] = max(0, segment["start"] - pad_sec)
                segment["end"] = min(len(audio_data) / sample_rate, segment["end"] + pad_sec)
                segment["duration"] = segment["end"] - segment["start"]
            
            # Merge overlapping segments
            segments = self._merge_overlapping_segments(segments)
            
            logger.info(f"Detected {len(segments)} speech segments using ML model")
            return segments
            
        except Exception as e:
            logger.error(f"Error in ML-based speech detection: {e}")
            # Fall back to energy-based approach
            logger.info("Falling back to energy-based speech detection")
            return self._detect_speech_energy(audio_data, sample_rate, progress_callback)
    
    def _detect_speech_energy(self, 
                             audio_data: np.ndarray, 
                             sample_rate: int,
                             progress_callback: Optional[Callable[[float], None]] = None) -> List[Dict]:
        """Detect speech segments using energy-based approach.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            progress_callback: Optional callback function for reporting progress (0-1)
            
        Returns:
            List[Dict]: List of speech segments with start/end times and confidence scores
        """
        # Calculate window sizes in samples
        window_size = int(self.settings["window_size_ms"] * sample_rate / 1000)
        smoothing_window = int(self.settings["smoothing_window_ms"] * sample_rate / 1000)
        min_speech_samples = int(self.settings["min_speech_duration_ms"] * sample_rate / 1000)
        min_silence_samples = int(self.settings["min_silence_duration_ms"] * sample_rate / 1000)
        speech_pad_samples = int(self.settings["speech_pad_ms"] * sample_rate / 1000)
        
        # Calculate energy
        energy = np.array([
            np.mean(audio_data[i:i+window_size]**2) 
            for i in range(0, len(audio_data) - window_size, window_size)
        ])
        
        # Normalize energy to 0-1 range
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        # Apply smoothing
        if smoothing_window > 1 and len(energy) > smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            energy = np.convolve(energy, kernel, mode='same')
        
        # Apply threshold
        is_speech = energy > self.settings["energy_threshold"]
        
        # Find speech segments
        segments = []
        in_speech = False
        speech_start = 0
        
        # Report progress at 25%
        if progress_callback:
            progress_callback(0.25)
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Speech start
                speech_start = i
                in_speech = True
            elif not speech and in_speech:
                # Speech end
                speech_end = i
                
                # Convert frame indices to time
                speech_start_time = speech_start * window_size / sample_rate
                speech_end_time = speech_end * window_size / sample_rate
                
                # Only add if duration meets minimum
                duration = speech_end_time - speech_start_time
                if duration * sample_rate >= min_speech_samples:
                    # Calculate confidence as normalized average energy in segment
                    segment_energy = np.mean(energy[speech_start:speech_end])
                    confidence = min(1.0, segment_energy * 2)  # Scale up for better confidence values
                    
                    # Add segment with padding
                    pad_time = speech_pad_samples / sample_rate
                    segment = {
                        "start": max(0, speech_start_time - pad_time),
                        "end": min(len(audio_data) / sample_rate, speech_end_time + pad_time),
                        "confidence": float(confidence)
                    }
                    segment["duration"] = segment["end"] - segment["start"]
                    segments.append(segment)
                
                in_speech = False
        
        # Handle case where audio ends during speech
        if in_speech:
            speech_end_time = len(is_speech) * window_size / sample_rate
            speech_start_time = speech_start * window_size / sample_rate
            
            # Only add if duration meets minimum
            duration = speech_end_time - speech_start_time
            if duration * sample_rate >= min_speech_samples:
                # Calculate confidence as normalized average energy in segment
                segment_energy = np.mean(energy[speech_start:])
                confidence = min(1.0, segment_energy * 2)  # Scale up for better confidence values
                
                # Add segment with padding
                pad_time = speech_pad_samples / sample_rate
                segment = {
                    "start": max(0, speech_start_time - pad_time),
                    "end": min(len(audio_data) / sample_rate, speech_end_time + pad_time),
                    "confidence": float(confidence)
                }
                segment["duration"] = segment["end"] - segment["start"]
                segments.append(segment)
        
        # Report progress at 50%
        if progress_callback:
            progress_callback(0.5)
        
        # Merge segments that are close together
        segments = self._merge_close_segments(segments, min_silence_samples / sample_rate)
        
        # Report progress at 75%
        if progress_callback:
            progress_callback(0.75)
        
        logger.info(f"Detected {len(segments)} speech segments using energy-based approach")
        
        # Report progress at 100%
        if progress_callback:
            progress_callback(1.0)
            
        return segments
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping speech segments.
        
        Args:
            segments: List of speech segments
            
        Returns:
            List[Dict]: Merged speech segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s["start"])
        
        # Merge overlapping segments
        merged = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            prev = merged[-1]
            
            # Check if segments overlap
            if segment["start"] <= prev["end"]:
                # Merge segments
                prev["end"] = max(prev["end"], segment["end"])
                prev["duration"] = prev["end"] - prev["start"]
                # Take max confidence
                prev["confidence"] = max(prev["confidence"], segment["confidence"])
            else:
                # Add as new segment
                merged.append(segment)
        
        return merged
    
    def _merge_close_segments(self, segments: List[Dict], max_gap: float) -> List[Dict]:
        """Merge segments that are close together.
        
        Args:
            segments: List of speech segments
            max_gap: Maximum gap between segments to merge (in seconds)
            
        Returns:
            List[Dict]: Merged speech segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s["start"])
        
        # Merge segments with small gaps
        merged = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            prev = merged[-1]
            
            # Check if gap is small enough to merge
            if segment["start"] - prev["end"] <= max_gap:
                # Merge segments
                prev["end"] = segment["end"]
                prev["duration"] = prev["end"] - prev["start"]
                # Take average confidence
                prev["confidence"] = (prev["confidence"] + segment["confidence"]) / 2
            else:
                # Add as new segment
                merged.append(segment)
        
        return merged
    
    def get_speech_mask(self, 
                       audio_data: np.ndarray, 
                       sample_rate: int,
                       segments: List[Dict]) -> np.ndarray:
        """Generate a binary mask for speech regions.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            segments: List of speech segments
            
        Returns:
            np.ndarray: Binary mask (1 for speech, 0 for non-speech)
        """
        # Initialize mask with zeros
        mask = np.zeros_like(audio_data)
        
        # Set speech regions to 1
        for segment in segments:
            start_sample = int(segment["start"] * sample_rate)
            end_sample = int(segment["end"] * sample_rate)
            
            # Ensure valid indices
            start_sample = max(0, start_sample)
            end_sample = min(len(mask), end_sample)
            
            if start_sample < end_sample:
                mask[start_sample:end_sample] = 1
        
        return mask
    
    def calculate_speech_statistics(self, segments: List[Dict], total_duration: float) -> Dict:
        """Calculate statistics about speech segments.
        
        Args:
            segments: List of speech segments
            total_duration: Total duration of the audio in seconds
            
        Returns:
            Dict: Statistics including speech percentage, count, avg duration, etc.
        """
        if not segments:
            return {
                "speech_percentage": 0.0,
                "speech_count": 0,
                "total_speech_duration": 0.0,
                "avg_speech_duration": 0.0,
                "max_speech_duration": 0.0,
                "min_speech_duration": 0.0,
                "avg_confidence": 0.0
            }
        
        # Calculate total speech duration
        total_speech = sum(s["duration"] for s in segments)
        
        # Calculate statistics
        stats = {
            "speech_percentage": (total_speech / total_duration) * 100 if total_duration > 0 else 0.0,
            "speech_count": len(segments),
            "total_speech_duration": total_speech,
            "avg_speech_duration": total_speech / len(segments),
            "max_speech_duration": max(s["duration"] for s in segments),
            "min_speech_duration": min(s["duration"] for s in segments),
            "avg_confidence": sum(s["confidence"] for s in segments) / len(segments)
        }
        
        return stats
    
    def visualize_speech_segments(self, 
                                 audio_data: np.ndarray, 
                                 sample_rate: int,
                                 segments: List[Dict]) -> np.ndarray:
        """Generate a visualization of speech segments.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            segments: List of speech segments
            
        Returns:
            np.ndarray: Visualization array (can be saved as image)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            import io
            
            # Create figure
            fig = Figure(figsize=(10, 4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Plot waveform
            time = np.arange(len(audio_data)) / sample_rate
            ax.plot(time, audio_data, color='gray', alpha=0.5)
            
            # Plot speech segments
            for segment in segments:
                ax.axvspan(segment["start"], segment["end"], 
                          alpha=0.3, color='green', 
                          label=f"Speech ({segment['confidence']:.2f})")
            
            # Set labels and title
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Speech Segments')
            
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            # Render figure to numpy array
            fig.tight_layout()
            canvas.draw()
            
            # Convert to numpy array
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            import matplotlib.image as mpimg
            img_arr = mpimg.imread(buf)
            
            plt.close(fig)
            
            return img_arr
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return np.zeros((400, 1000, 3))  # Return empty image 