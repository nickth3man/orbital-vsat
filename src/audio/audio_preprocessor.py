"""
Audio preprocessing module for VSAT.

This module provides functionality for enhancing audio quality before main processing.
"""

import logging
import numpy as np
import librosa
import scipy.signal as signal
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

from src.utils.error_handler import AudioError, ErrorSeverity

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Class for preprocessing audio to improve quality before main processing."""
    
    # Preset configurations for common recording environments
    PRESETS = {
        "default": {
            "noise_reduction": {"threshold": 0.015, "reduction_factor": 0.7},
            "normalization": {"target_level": -23.0, "headroom": 3.0},
            "equalization": {"profile": "flat"}
        },
        "conference_room": {
            "noise_reduction": {"threshold": 0.02, "reduction_factor": 0.8},
            "normalization": {"target_level": -20.0, "headroom": 2.0},
            "equalization": {"profile": "speech_enhance"}
        },
        "interview": {
            "noise_reduction": {"threshold": 0.01, "reduction_factor": 0.6},
            "normalization": {"target_level": -18.0, "headroom": 2.5},
            "equalization": {"profile": "voice"}
        },
        "noisy_environment": {
            "noise_reduction": {"threshold": 0.03, "reduction_factor": 0.9},
            "normalization": {"target_level": -20.0, "headroom": 3.0},
            "equalization": {"profile": "noise_reduction"}
        },
        "music_background": {
            "noise_reduction": {"threshold": 0.025, "reduction_factor": 0.7},
            "normalization": {"target_level": -22.0, "headroom": 3.0},
            "equalization": {"profile": "voice_over_music"}
        }
    }
    
    # Equalization profiles with frequency bands and gain values
    EQ_PROFILES = {
        "flat": [
            (0, 20000, 0.0)  # (start_freq, end_freq, gain_db)
        ],
        "speech_enhance": [
            (0, 300, -3.0),      # Reduce low frequencies
            (300, 3000, 3.0),    # Boost speech frequencies
            (3000, 20000, -1.0)  # Slightly reduce high frequencies
        ],
        "voice": [
            (0, 200, -4.0),      # Reduce rumble
            (200, 800, 1.0),     # Slight boost to lower voice frequencies
            (800, 3000, 4.0),    # Boost speech intelligibility
            (3000, 6000, 2.0),   # Add presence
            (6000, 20000, -2.0)  # Reduce hiss
        ],
        "noise_reduction": [
            (0, 300, -6.0),      # Reduce low frequency noise
            (300, 2500, 3.0),    # Boost speech
            (2500, 5000, 1.0),   # Maintain some presence
            (5000, 20000, -8.0)  # Significantly reduce high frequency noise
        ],
        "voice_over_music": [
            (0, 200, -3.0),      # Reduce bass
            (200, 500, -1.0),    # Slightly reduce low mids
            (500, 3000, 5.0),    # Significantly boost voice frequencies
            (3000, 6000, 2.0),   # Add presence
            (6000, 20000, -2.0)  # Reduce high frequencies
        ]
    }
    
    def __init__(self):
        """Initialize the audio preprocessor."""
        logger.info("Initializing audio preprocessor")
    
    def preprocess_audio(self, 
                        audio_data: np.ndarray, 
                        sample_rate: int,
                        preset: str = "default",
                        custom_settings: Optional[Dict[str, Any]] = None,
                        progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """Apply preprocessing to improve audio quality.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            preset: Preset name from PRESETS or "custom"
            custom_settings: Custom settings to use if preset is "custom"
            progress_callback: Optional callback function for progress updates
            
        Returns:
            np.ndarray: Preprocessed audio data
            
        Raises:
            AudioError: If there's an error during preprocessing
        """
        try:
            # Get settings from preset or custom settings
            if preset == "custom" and custom_settings:
                settings = custom_settings
            elif preset in self.PRESETS:
                settings = self.PRESETS[preset]
            else:
                raise AudioError(
                    f"Invalid preset: {preset}",
                    ErrorSeverity.ERROR,
                    {"preset": preset, "available_presets": list(self.PRESETS.keys())}
                )
            
            # Make a copy of the audio data to avoid modifying the original
            processed_audio = audio_data.copy()
            
            # Apply noise reduction if enabled
            if "noise_reduction" in settings and settings["noise_reduction"]:
                if progress_callback:
                    progress_callback(0.2, "Applying noise reduction...")
                
                processed_audio = self.apply_noise_reduction(
                    processed_audio, 
                    sample_rate,
                    threshold=settings["noise_reduction"].get("threshold", 0.015),
                    reduction_factor=settings["noise_reduction"].get("reduction_factor", 0.7)
                )
            
            # Apply normalization if enabled
            if "normalization" in settings and settings["normalization"]:
                if progress_callback:
                    progress_callback(0.5, "Applying audio normalization...")
                
                processed_audio = self.apply_normalization(
                    processed_audio,
                    target_level=settings["normalization"].get("target_level", -23.0),
                    headroom=settings["normalization"].get("headroom", 3.0)
                )
            
            # Apply equalization if enabled
            if "equalization" in settings and settings["equalization"]:
                if progress_callback:
                    progress_callback(0.8, "Applying equalization...")
                
                profile = settings["equalization"].get("profile", "flat")
                processed_audio = self.apply_equalization(
                    processed_audio,
                    sample_rate,
                    profile=profile
                )
            
            if progress_callback:
                progress_callback(1.0, "Preprocessing complete")
            
            return processed_audio
            
        except Exception as e:
            if not isinstance(e, AudioError):
                raise AudioError(
                    f"Error during audio preprocessing: {str(e)}",
                    ErrorSeverity.ERROR,
                    {"preset": preset, "error": str(e)}
                ) from e
            raise
    
    def apply_noise_reduction(self, 
                             audio_data: np.ndarray, 
                             sample_rate: int,
                             threshold: float = 0.015,
                             reduction_factor: float = 0.7) -> np.ndarray:
        """Apply spectral subtraction noise reduction.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            threshold: Noise threshold (0.0-1.0)
            reduction_factor: Noise reduction strength (0.0-1.0)
            
        Returns:
            np.ndarray: Noise-reduced audio data
        """
        logger.info(f"Applying noise reduction with threshold={threshold}, reduction_factor={reduction_factor}")
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_mono = audio_data.mean(axis=1)
        else:
            audio_mono = audio_data.copy()
        
        # Parameters for STFT
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        stft = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = librosa.magphase(stft)
        
        # Estimate noise profile from the first 500ms of audio
        noise_frames = int(0.5 * sample_rate / hop_length)
        noise_frames = min(noise_frames, magnitude.shape[1])
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        threshold_value = threshold * np.max(magnitude)
        magnitude_reduced = np.maximum(
            magnitude - noise_profile * reduction_factor,
            threshold_value
        )
        
        # Reconstruct signal
        stft_reduced = magnitude_reduced * phase
        audio_reduced = librosa.istft(stft_reduced, hop_length=hop_length)
        
        # Resize to original length
        if len(audio_reduced) < len(audio_mono):
            audio_reduced = np.pad(audio_reduced, (0, len(audio_mono) - len(audio_reduced)))
        else:
            audio_reduced = audio_reduced[:len(audio_mono)]
        
        # If original was stereo, convert back to stereo
        if audio_data.ndim > 1:
            # Create a scaling factor for each channel
            scaling = audio_reduced / (audio_mono + 1e-10)  # Avoid division by zero
            result = np.zeros_like(audio_data)
            for i in range(audio_data.shape[1]):
                result[:, i] = audio_data[:, i] * scaling
            return result
        
        return audio_reduced
    
    def apply_normalization(self, 
                           audio_data: np.ndarray,
                           target_level: float = -23.0,
                           headroom: float = 3.0) -> np.ndarray:
        """Normalize audio to a target loudness level.
        
        Args:
            audio_data: Audio data as numpy array
            target_level: Target loudness level in dB LUFS
            headroom: Headroom to leave in dB
            
        Returns:
            np.ndarray: Normalized audio data
        """
        logger.info(f"Applying normalization to target level {target_level}dB with {headroom}dB headroom")
        
        # Calculate current peak level
        peak_level = np.max(np.abs(audio_data))
        
        # Calculate current RMS level (as a simple proxy for loudness)
        rms_level = np.sqrt(np.mean(audio_data**2))
        
        # Convert to dB
        if rms_level > 0:
            current_db = 20 * np.log10(rms_level)
        else:
            current_db = -100.0
        
        # Calculate gain needed
        gain_db = target_level - current_db
        
        # Apply headroom limit
        max_gain_db = -headroom - (20 * np.log10(peak_level) if peak_level > 0 else -100.0)
        gain_db = min(gain_db, max_gain_db)
        
        # Convert to linear gain
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Apply gain
        normalized_audio = audio_data * gain_linear
        
        return normalized_audio
    
    def apply_equalization(self,
                          audio_data: np.ndarray,
                          sample_rate: int,
                          profile: str = "flat") -> np.ndarray:
        """Apply equalization to enhance specific frequency ranges.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            profile: Equalization profile name from EQ_PROFILES
            
        Returns:
            np.ndarray: Equalized audio data
            
        Raises:
            AudioError: If the profile is invalid
        """
        if profile not in self.EQ_PROFILES:
            raise AudioError(
                f"Invalid equalization profile: {profile}",
                ErrorSeverity.ERROR,
                {"profile": profile, "available_profiles": list(self.EQ_PROFILES.keys())}
            )
        
        logger.info(f"Applying equalization with profile '{profile}'")
        
        # Get the EQ bands for the selected profile
        eq_bands = self.EQ_PROFILES[profile]
        
        # If profile is flat, return the original audio
        if profile == "flat":
            return audio_data
        
        # Convert to mono if stereo for processing
        is_stereo = audio_data.ndim > 1
        if is_stereo:
            channels = []
            for i in range(audio_data.shape[1]):
                channel = audio_data[:, i].copy()
                # Process each channel separately
                for start_freq, end_freq, gain_db in eq_bands:
                    channel = self._apply_band_gain(channel, sample_rate, start_freq, end_freq, gain_db)
                channels.append(channel)
            
            # Recombine channels
            equalized_audio = np.column_stack(channels)
        else:
            equalized_audio = audio_data.copy()
            for start_freq, end_freq, gain_db in eq_bands:
                equalized_audio = self._apply_band_gain(equalized_audio, sample_rate, start_freq, end_freq, gain_db)
        
        return equalized_audio
    
    def _apply_band_gain(self,
                        audio_data: np.ndarray,
                        sample_rate: int,
                        start_freq: float,
                        end_freq: float,
                        gain_db: float) -> np.ndarray:
        """Apply gain to a specific frequency band.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            start_freq: Start frequency of the band in Hz
            end_freq: End frequency of the band in Hz
            gain_db: Gain to apply in dB
            
        Returns:
            np.ndarray: Audio data with gain applied to the specified band
        """
        # Convert frequencies to normalized frequency (0 to 1)
        nyquist = sample_rate / 2
        start_norm = start_freq / nyquist
        end_norm = end_freq / nyquist
        
        # Design bandpass filter
        if start_freq <= 0:
            # Low-pass filter
            b, a = signal.butter(4, end_norm, btype='lowpass')
        elif end_freq >= nyquist:
            # High-pass filter
            b, a = signal.butter(4, start_norm, btype='highpass')
        else:
            # Band-pass filter
            b, a = signal.butter(4, [start_norm, end_norm], btype='bandpass')
        
        # Apply filter to get the band
        band = signal.filtfilt(b, a, audio_data)
        
        # Apply gain to the band
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Add the modified band back to the original signal
        return audio_data + (band * gain_linear - band)
    
    def batch_preprocess(self,
                        file_paths: List[str],
                        output_dir: str,
                        preset: str = "default",
                        custom_settings: Optional[Dict[str, Any]] = None,
                        progress_callback: Optional[Callable[[float, str, str], None]] = None) -> List[str]:
        """Preprocess multiple audio files in batch.
        
        Args:
            file_paths: List of audio file paths
            output_dir: Directory to save preprocessed files
            preset: Preset name from PRESETS or "custom"
            custom_settings: Custom settings to use if preset is "custom"
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List[str]: List of paths to preprocessed files
            
        Raises:
            AudioError: If there's an error during preprocessing
        """
        from src.audio.file_handler import AudioFileHandler
        import os
        from pathlib import Path
        
        file_handler = AudioFileHandler()
        output_paths = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Update overall progress
                if progress_callback:
                    overall_progress = i / len(file_paths)
                    progress_callback(overall_progress, f"Processing file {i+1}/{len(file_paths)}", file_path)
                
                # Define a file-specific progress callback
                def file_progress(prog, status):
                    if progress_callback:
                        # Scale the file progress to a small portion of the overall progress
                        file_portion = 1.0 / len(file_paths)
                        overall_prog = i / len(file_paths) + (prog * file_portion)
                        progress_callback(overall_prog, status, file_path)
                
                # Load audio file
                audio_data, sample_rate, metadata = file_handler.load_audio(file_path)
                
                # Preprocess audio
                processed_audio = self.preprocess_audio(
                    audio_data,
                    sample_rate,
                    preset=preset,
                    custom_settings=custom_settings,
                    progress_callback=file_progress
                )
                
                # Create output filename
                file_name = os.path.basename(file_path)
                base_name, ext = os.path.splitext(file_name)
                output_file = os.path.join(output_dir, f"{base_name}_preprocessed{ext}")
                
                # Ensure output directory exists
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Save preprocessed audio
                file_handler.save_audio(output_file, processed_audio, sample_rate, metadata)
                
                output_paths.append(output_file)
                
            except Exception as e:
                logger.error(f"Error preprocessing file {file_path}: {str(e)}")
                # Continue with next file instead of stopping the batch
                continue
        
        return output_paths
    
    def get_available_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available preprocessing presets.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of preset names and their settings
        """
        return self.PRESETS.copy()
    
    def get_available_eq_profiles(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """Get available equalization profiles.
        
        Returns:
            Dict[str, List[Tuple[float, float, float]]]: Dictionary of profile names and their bands
        """
        return self.EQ_PROFILES.copy() 