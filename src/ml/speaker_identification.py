"""
Speaker identification module for VSAT.

This module provides functionality for speaker identification using ECAPA-TDNN embeddings.
It handles voice print generation, storage, and matching of speakers across recordings.
"""

import os
import logging
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

# Import database models
from src.database.models import Speaker

# Import audio file handler
from src.audio.file_handler import AudioFileHandler
from src.ml.voice_print_processor import VoicePrintProcessor

logger = logging.getLogger(__name__)

class SpeakerIdentifier:
    """Class for speaker identification using ECAPA-TDNN embeddings."""
    
    def __init__(self, 
                 auth_token: Optional[str] = None,
                 device: str = "cpu",
                 download_root: Optional[str] = None,
                 similarity_threshold: float = 0.75):
        """Initialize the speaker identifier.
        
        Args:
            auth_token: HuggingFace authentication token
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
            similarity_threshold: Threshold for speaker similarity (0-1)
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        # Set default download root if not provided
        if download_root is None:
            download_root = str(Path.home() / '.vsat' / 'models' / 'speaker_id')
        
        # Create download directory if it doesn't exist
        os.makedirs(download_root, exist_ok=True)
        
        # Initialize voice print processor
        self.voice_print_processor = VoicePrintProcessor(auth_token, device, download_root)
        
        # Initialize audio file handler
        self.file_handler = AudioFileHandler()
        
        logger.info(f"Speaker identifier initialized (device: {device}, threshold: {similarity_threshold})")
    
    def generate_voice_print(self, audio: Union[str, np.ndarray, torch.Tensor], 
                            sample_rate: Optional[int] = None) -> np.ndarray:
        """Generate a voice print embedding from audio.
        
        Args:
            audio: Audio file path, numpy array, or tensor
            sample_rate: Sample rate of the audio (required if audio is array/tensor)
            
        Returns:
            np.ndarray: Speaker embedding vector
        """
        return self.voice_print_processor.generate_voice_print(audio, sample_rate)
    
    def compare_voice_prints(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two voice print embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            float: Similarity score (0-1)
        """
        return self.voice_print_processor.compare_voice_prints(embedding1, embedding2)
    
    def find_matching_speaker(self, embedding: np.ndarray, 
                             speakers: List[Speaker]) -> Tuple[Optional[Speaker], float]:
        """Find a matching speaker from a list of speakers.
        
        Args:
            embedding: Speaker embedding to match
            speakers: List of speakers to compare against
            
        Returns:
            Tuple[Optional[Speaker], float]: Matching speaker and similarity score, or (None, 0.0)
        """
        if not speakers:
            logger.debug("No speakers provided for matching")
            return None, 0.0
        
        # Extract embeddings from speakers
        speaker_embeddings = []
        valid_speakers = []
        
        for speaker in speakers:
            if speaker.voice_print:
                try:
                    # Load embedding from voice print
                    embedding_data = pickle.loads(speaker.voice_print)
                    speaker_embeddings.append(embedding_data)
                    valid_speakers.append(speaker)
                except Exception as e:
                    logger.error(f"Error loading voice print for speaker {speaker.id}: {e}")
        
        if not speaker_embeddings:
            logger.debug("No valid voice prints found in speakers")
            return None, 0.0
        
        # Compare embedding with all speaker embeddings
        similarities = [self.compare_voice_prints(embedding, spk_emb) for spk_emb in speaker_embeddings]
        
        # Find the best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Check if the best match is above the threshold
        if best_score >= self.similarity_threshold:
            logger.debug(f"Found matching speaker with score {best_score:.4f}")
            return valid_speakers[best_idx], best_score
        else:
            logger.debug(f"No matching speaker found (best score: {best_score:.4f})")
            return None, best_score
    
    def update_speaker_voice_print(self, speaker: Speaker, 
                                  new_embedding: np.ndarray, 
                                  learning_rate: float = 0.3) -> np.ndarray:
        """Update a speaker's voice print with a new embedding.
        
        Args:
            speaker: Speaker to update
            new_embedding: New embedding to incorporate
            learning_rate: Weight of the new embedding (0-1)
            
        Returns:
            np.ndarray: Updated speaker embedding
        """
        return self.voice_print_processor.update_speaker_voice_print(
            speaker, new_embedding, learning_rate
        )
    
    def process_diarization_results(self, audio_path: str, 
                                   diarization_results: Dict[str, Any],
                                   existing_speakers: List[Speaker]) -> Dict[str, Any]:
        """Process diarization results to identify speakers.
        
        Args:
            audio_path: Path to the audio file
            diarization_results: Results from diarization
            existing_speakers: List of existing speakers to match against
            
        Returns:
            Dict[str, Any]: Processed results with identified speakers
        """
        # Load audio file
        audio_data, sample_rate, _ = self.file_handler.load_audio(audio_path)
        
        # Process each speaker in the diarization results
        processed_results = diarization_results.copy()
        
        # Track new speakers
        new_speakers = {}
        
        # Process each segment
        for i, segment in enumerate(processed_results.get('segments', [])):
            # Get speaker label from diarization
            speaker_label = segment.get('speaker')
            
            if not speaker_label:
                continue
                
            # Check if we've already processed this speaker
            if speaker_label in new_speakers:
                # Use the already identified speaker
                segment['speaker'] = new_speakers[speaker_label]
                continue
                
            # Extract audio for this segment
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            # Ensure valid range
            if start_sample >= end_sample or start_sample >= len(audio_data) or end_sample > len(audio_data):
                logger.warning(f"Invalid segment range: {segment['start']} - {segment['end']}")
                continue
                
            segment_audio = audio_data[start_sample:end_sample]
            
            # Generate voice print for this segment
            try:
                embedding = self.generate_voice_print(segment_audio, sample_rate)
                
                # Find matching speaker
                matching_speaker, similarity = self.find_matching_speaker(embedding, existing_speakers)
                
                if matching_speaker:
                    # Use existing speaker
                    segment['speaker'] = matching_speaker.id
                    segment['speaker_name'] = matching_speaker.name
                    segment['speaker_similarity'] = float(similarity)
                    
                    # Update speaker's voice print
                    self.update_speaker_voice_print(matching_speaker, embedding)
                    
                    # Remember this mapping
                    new_speakers[speaker_label] = matching_speaker.id
                else:
                    # Keep the original speaker label
                    pass
                    
            except Exception as e:
                logger.error(f"Error processing segment {i}: {e}")
        
        return processed_results 