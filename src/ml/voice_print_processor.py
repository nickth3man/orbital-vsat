"""
Voice print processing module for VSAT.

This module provides functionality for generating, comparing, and updating voice print embeddings.
"""

import os
import logging
import numpy as np
import torch
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

# Import database models
from src.database.models import Speaker

logger = logging.getLogger(__name__)

class VoicePrintProcessor:
    """Class for processing voice print embeddings."""
    
    def __init__(self, 
                 auth_token: Optional[str] = None,
                 device: str = "cpu",
                 download_root: Optional[str] = None):
        """Initialize the voice print processor.
        
        Args:
            auth_token: HuggingFace authentication token
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
        """
        self.device = device
        
        logger.info(f"Initializing voice print processor on {device}")
        
        try:
            # Initialize the ECAPA-TDNN model from SpeechBrain
            from speechbrain.pretrained import EncoderClassifier
            
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=download_root,
                run_opts={"device": device}
            )
            
            logger.info("Voice print processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing voice print processor: {e}")
            raise
    
    def generate_voice_print(self, audio: Union[str, np.ndarray, torch.Tensor], 
                            sample_rate: Optional[int] = None) -> np.ndarray:
        """Generate a voice print embedding from audio.
        
        Args:
            audio: Path to audio file, numpy array, or torch tensor
            sample_rate: Sample rate of the audio (required if audio is numpy array)
            
        Returns:
            np.ndarray: Voice print embedding
            
        Raises:
            ValueError: If sample_rate is not provided for numpy array input
        """
        try:
            # Handle different input types
            if isinstance(audio, str):
                # Audio is a file path
                signal, fs = self.model.load_audio(audio)
                
            elif isinstance(audio, np.ndarray):
                # Audio is a numpy array
                if sample_rate is None:
                    raise ValueError("Sample rate must be provided when audio is a numpy array")
                
                # Convert to torch tensor
                signal = torch.tensor(audio, dtype=torch.float32)
                if signal.ndim == 1:
                    signal = signal.unsqueeze(0)
                fs = sample_rate
                
            elif isinstance(audio, torch.Tensor):
                # Audio is already a torch tensor
                signal = audio
                if signal.ndim == 1:
                    signal = signal.unsqueeze(0)
                
                if sample_rate is None:
                    raise ValueError("Sample rate must be provided when audio is a torch tensor")
                fs = sample_rate
                
            else:
                raise TypeError(f"Unsupported audio type: {type(audio)}")
            
            # Generate embedding
            embeddings = self.model.encode_batch(signal)
            
            # Convert to numpy and return the first embedding
            embedding = embeddings[0].cpu().numpy()
            
            logger.info(f"Generated voice print with shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating voice print: {e}")
            raise
    
    def compare_voice_prints(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two voice print embeddings and return similarity score.
        
        Args:
            embedding1: First voice print embedding
            embedding2: Second voice print embedding
            
        Returns:
            float: Similarity score (0-1, higher is more similar)
        """
        try:
            # Reshape embeddings if needed
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing voice prints: {e}")
            raise
    
    def update_speaker_voice_print(self, speaker: Speaker, 
                                  new_embedding: np.ndarray, 
                                  learning_rate: float = 0.3) -> np.ndarray:
        """Update a speaker's voice print with a new embedding.
        
        Args:
            speaker: Speaker object to update
            new_embedding: New voice print embedding
            learning_rate: Weight for the new embedding (0-1)
            
        Returns:
            np.ndarray: Updated voice print embedding
        """
        try:
            # If the speaker doesn't have a voice print yet, use the new one
            if not speaker.voice_print:
                updated_embedding = new_embedding
                logger.info(f"Created initial voice print for speaker {speaker.name}")
            else:
                # Load the existing voice print
                existing_embedding = pickle.loads(speaker.voice_print)
                
                # Combine the embeddings with the specified learning rate
                updated_embedding = (1 - learning_rate) * existing_embedding + learning_rate * new_embedding
                
                # Normalize the embedding
                norm = np.linalg.norm(updated_embedding)
                if norm > 0:
                    updated_embedding = updated_embedding / norm
                
                logger.info(f"Updated voice print for speaker {speaker.name} with learning rate {learning_rate}")
            
            return updated_embedding
            
        except Exception as e:
            logger.error(f"Error updating speaker voice print: {e}")
            raise 