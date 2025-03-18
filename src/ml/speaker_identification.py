"""
Speaker identification module for VSAT.

This module provides functionality for speaker identification using ECAPA-TDNN
embeddings. It handles voice print generation, storage, and matching of speakers
across recordings.
"""

import os
import logging
import numpy as np
import torch
import pickle
import traceback
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity

# Import database models
from ..database.models import Speaker

# Import audio file handler
from ..audio.file_handler import AudioFileHandler
from src.ml.voice_print_processor import VoicePrintProcessor
from ..ml.error_handling import (
    SpeakerIdentificationError, ModelLoadError, ResourceExhaustionError
)

logger = logging.getLogger(__name__)

# Define common speaker identification errors and their recovery strategies
SPEAKER_ID_ERROR_TYPES = {
    "model_load": "Failed to load speaker identification model",
    "embedding_generation": "Failed to generate voice print embedding",
    "similarity_calculation": "Failed to calculate similarity between voice prints",
    "speaker_matching": "Failed to find matching speaker",
    "voice_print_update": "Failed to update speaker voice print",
    "database_access": "Failed to access speaker database",
    "audio_quality": "Poor audio quality affecting speaker identification"
}


@dataclass
class SpeakerMatch:
    """Represents a match between a voice sample and a known speaker.

    This class contains information about a speaker match, including the matched
    speaker, confidence score, and additional metadata about the match.
    """

    speaker: Optional[Speaker] = None
    confidence: float = 0.0
    timestamp: Optional[float] = None
    audio_segment: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the SpeakerMatch object after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the SpeakerMatch object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the SpeakerMatch
        """
        result = {
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata.copy()
        }

        # Add speaker information if available
        if self.speaker:
            result["speaker"] = {
                "id": self.speaker.id,
                "name": self.speaker.name,
                "notes": getattr(self.speaker, "notes", None)
            }

        # Add audio segment information if available
        if self.audio_segment:
            result["audio_segment"] = self.audio_segment.copy()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpeakerMatch':
        """Create a SpeakerMatch object from a dictionary.

        Args:
            data: Dictionary containing SpeakerMatch data

        Returns:
            SpeakerMatch: New SpeakerMatch object

        Raises:
            ValueError: If the dictionary is missing required fields or has
                invalid values
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        # Extract speaker information if available
        speaker = None
        if "speaker" in data and data["speaker"]:
            # This is a simplified approach - in a real application, you might
            # want to fetch the actual Speaker object from the database
            speaker_data = data["speaker"]
            speaker = Speaker(
                id=speaker_data.get("id"),
                name=speaker_data.get("name"),
                notes=speaker_data.get("notes")
            )

        return cls(
            speaker=speaker,
            confidence=data.get("confidence", 0.0),
            timestamp=data.get("timestamp"),
            audio_segment=data.get("audio_segment"),
            metadata=data.get("metadata", {})
        )

    def is_match(self, threshold: float = 0.75) -> bool:
        """Check if the match exceeds the confidence threshold.

        Args:
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            bool: True if the confidence exceeds the threshold
        """
        return self.confidence >= threshold

    def __str__(self) -> str:
        """Return a string representation of the SpeakerMatch."""
        speaker_name = self.speaker.name if self.speaker else "Unknown"
        return (
            f"SpeakerMatch(speaker={speaker_name}, "
            f"confidence={self.confidence:.2f})"
        )


class SpeakerIdentifier:
    """Class for speaker identification using ECAPA-TDNN embeddings."""

    def __init__(
        self,
        auth_token: Optional[str] = None,
        device: str = "cpu",
        download_root: Optional[str] = None,
        similarity_threshold: float = 0.75,
        timeout: int = 300
    ):
        """Initialize the speaker identifier.

        Args:
            auth_token: HuggingFace authentication token
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
            similarity_threshold: Threshold for speaker similarity (0-1)
            timeout: Maximum time in seconds to allow for processing

        Raises:
            ModelLoadError: If the speaker identification model fails to load
            ResourceExhaustionError: If system resources are exhausted during
                initialization
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.timeout = timeout
        self.max_retries = 3

        # Set default download root if not provided
        if download_root is None:
            download_root = str(
                Path.home() / '.vsat' / 'models' / 'speaker_id'
            )

        # Create download directory if it doesn't exist
        os.makedirs(download_root, exist_ok=True)

        # Initialize voice print processor
        try:
            self.voice_print_processor = VoicePrintProcessor(
                auth_token, device, download_root
            )
        except Exception as e:
            error_type = type(e).__name__
            error_context = {
                "device": device,
                "error_type": error_type,
                "original_error": str(e),
                "stack_trace": traceback.format_exc()
            }

            if "memory" in str(e).lower():
                raise ResourceExhaustionError(
                    "Out of memory during speaker identification model "
                    "initialization",
                    "memory",
                    error_context
                )
            elif "cuda" in str(e).lower():
                error_context["suggestion"] = "Use CPU device instead"
                raise ModelLoadError(
                    "CUDA error during speaker identification model "
                    f"initialization: {str(e)}",
                    "speaker-identification-ecapa-tdnn",
                    error_context
                )
            elif "auth" in str(e).lower() or "token" in str(e).lower():
                error_context["suggestion"] = "Check your HuggingFace auth token"
                raise ModelLoadError(
                    "Authentication failed for speaker identification model: "
                    f"{str(e)}",
                    "speaker-identification-ecapa-tdnn",
                    error_context
                )
            else:
                raise ModelLoadError(
                    "Failed to initialize speaker identification model: "
                    f"{str(e)}",
                    "speaker-identification-ecapa-tdnn",
                    error_context
                )

        # Initialize audio file handler
        self.file_handler = AudioFileHandler()

        logger.info(
            f"Speaker identifier initialized (device: {device}, "
            f"threshold: {similarity_threshold})"
        )

    def generate_voice_print(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Generate a voice print embedding from audio.

        Args:
            audio: Audio file path, numpy array, or tensor
            sample_rate: Sample rate of the audio (required if audio is
                array/tensor)

        Returns:
            np.ndarray: Speaker embedding vector

        Raises:
            SpeakerIdentificationError: If voice print generation fails
            ResourceExhaustionError: If system resources are exhausted
            ValueError: If the input is invalid
            FileNotFoundError: If the audio file does not exist
        """
        start_time = time.time()

        try:
            # Check if processing time exceeds timeout
            if time.time() - start_time > self.timeout:
                raise SpeakerIdentificationError(
                    f"Voice print generation timeout after {self.timeout} "
                    "seconds",
                    {
                        "error_type": "processing_timeout", 
                        "timeout": self.timeout
                    }
                )

            # Validate input audio
            if isinstance(audio, str):
                if not os.path.exists(audio):
                    raise FileNotFoundError(f"Audio file not found: {audio}")
            elif sample_rate is None:
                raise ValueError(
                    "Sample rate must be provided when audio is not a file path"
                )

            # Call voice print processor
            return self.voice_print_processor.generate_voice_print(
                audio, sample_rate
            )

        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"CUDA out of memory during voice print generation: {str(e)}"
            )
            raise ResourceExhaustionError(
                "CUDA out of memory during voice print generation. "
                "Try using a CPU device or reducing audio length.",
                "GPU memory",
                {
                    "device": self.device,
                    "original_error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )

        except (FileNotFoundError, ValueError):
            # Re-raise these as they are input validation errors
            raise

        except Exception as e:
            logger.error(f"Voice print generation failed: {str(e)}")
            error_type = type(e).__name__
            error_context = {
                "error_type": error_type,
                "original_error": str(e),
                "stack_trace": traceback.format_exc(),
                "processing_time": time.time() - start_time
            }

            # Add audio info if available
            if isinstance(audio, np.ndarray) and sample_rate is not None:
                error_context["audio_shape"] = audio.shape
                error_context["sample_rate"] = sample_rate
                error_context["audio_duration"] = len(audio) / sample_rate
            elif isinstance(audio, str):
                error_context["audio_file"] = audio

            raise SpeakerIdentificationError(
                f"Failed to generate voice print: {str(e)}",
                {"error_type": "embedding_generation", **error_context}
            )

    def compare_voice_prints(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compare two voice print embeddings.

        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding

        Returns:
            float: Similarity score (0-1)

        Raises:
            SpeakerIdentificationError: If comparison fails
            ValueError: If embeddings are invalid
        """
        try:
            # Validate embeddings
            if (not isinstance(embedding1, np.ndarray) or 
                    not isinstance(embedding2, np.ndarray)):
                raise ValueError("Embeddings must be numpy arrays")

            if embedding1.shape != embedding2.shape:
                raise ValueError(
                    f"Embedding shapes do not match: "
                    f"{embedding1.shape} vs {embedding2.shape}"
                )

            return self.voice_print_processor.compare_voice_prints(
                embedding1, embedding2
            )

        except ValueError:
            # Re-raise value errors as they are input validation errors
            raise

        except Exception as e:
            logger.error(f"Voice print comparison failed: {str(e)}")
            error_context = {
                "error_type": type(e).__name__,
                "original_error": str(e),
                "stack_trace": traceback.format_exc(),
                "embedding1_shape": getattr(embedding1, "shape", None),
                "embedding2_shape": getattr(embedding2, "shape", None)
            }

            raise SpeakerIdentificationError(
                f"Failed to compare voice prints: {str(e)}",
                {"error_type": "similarity_calculation", **error_context}
            )

    def find_matching_speaker(
        self,
        embedding: np.ndarray,
        speakers: List[Speaker]
    ) -> Tuple[Optional[Speaker], float]:
        """Find a matching speaker from a list of speakers.

        Args:
            embedding: Speaker embedding to match
            speakers: List of speakers to compare against

        Returns:
            Tuple[Optional[Speaker], float]: Matching speaker and similarity
                score, or (None, 0.0)

        Raises:
            SpeakerIdentificationError: If matching fails
            ValueError: If the embedding is invalid
        """
        if not speakers:
            logger.debug("No speakers provided for matching")
            return None, 0.0

        try:
            # Validate embedding
            if not isinstance(embedding, np.ndarray):
                raise ValueError("Embedding must be a numpy array")

            # Extract embeddings from speakers
            speaker_embeddings = []
            valid_speakers = []

            for speaker in speakers:
                if (hasattr(speaker, 'voice_print') and 
                        speaker.voice_print is not None):
                    try:
                        # Load voice print from bytes
                        voice_print = pickle.loads(speaker.voice_print)
                        speaker_embeddings.append(voice_print)
                        valid_speakers.append(speaker)
                    except Exception as e:
                        logger.warning(
                            f"Could not load voice print for speaker "
                            f"{speaker.id}: {e}"
                        )

            if not speaker_embeddings:
                logger.debug("No valid voice prints found in speakers list")
                return None, 0.0

            # Stack embeddings and compute similarity
            embedding_matrix = np.vstack(speaker_embeddings)
            similarities = cosine_similarity([embedding], embedding_matrix)[0]

            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            # Check if score exceeds threshold
            if best_score >= self.similarity_threshold:
                return valid_speakers[best_idx], best_score
            else:
                return None, best_score

        except ValueError:
            # Re-raise value errors as they are input validation errors
            raise

        except Exception as e:
            logger.error(f"Speaker matching failed: {str(e)}")
            error_context = {
                "error_type": type(e).__name__,
                "original_error": str(e),
                "stack_trace": traceback.format_exc(),
                "embedding_shape": getattr(embedding, "shape", None),
                "num_speakers": len(speakers),
                "similarity_threshold": self.similarity_threshold
            }

            raise SpeakerIdentificationError(
                f"Failed to find matching speaker: {str(e)}",
                {"error_type": "speaker_matching", **error_context}
            )

    def update_speaker_voice_print(
        self,
        speaker: Speaker,
        new_embedding: np.ndarray,
        learning_rate: float = 0.3
    ) -> np.ndarray:
        """Update a speaker's voice print with a new embedding.

        Args:
            speaker: Speaker to update
            new_embedding: New voice print embedding
            learning_rate: Weight of new embedding (0-1)

        Returns:
            np.ndarray: Updated voice print

        Raises:
            SpeakerIdentificationError: If update fails
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(new_embedding, np.ndarray):
                raise ValueError("New embedding must be a numpy array")

            if learning_rate < 0 or learning_rate > 1:
                raise ValueError(
                    f"Learning rate must be between 0 and 1, got {learning_rate}"
                )

            # Load current voice print
            if (hasattr(speaker, 'voice_print') and 
                    speaker.voice_print is not None):
                current_embedding = pickle.loads(speaker.voice_print)
            else:
                logger.info(
                    f"No existing voice print for speaker {speaker.id}, "
                    f"using new embedding"
                )
                return new_embedding

            # Validate embedding shapes
            if current_embedding.shape != new_embedding.shape:
                raise ValueError(
                    f"Embedding shapes do not match: "
                    f"{current_embedding.shape} vs {new_embedding.shape}"
                )

            # Update with weighted average
            updated_embedding = (
                (1 - learning_rate) * current_embedding +
                learning_rate * new_embedding
            )

            # Normalize
            updated_embedding = updated_embedding / np.linalg.norm(
                updated_embedding
            )

            return updated_embedding

        except ValueError:
            # Re-raise value errors as they are input validation errors
            raise

        except Exception as e:
            logger.error(f"Speaker voice print update failed: {str(e)}")
            error_context = {
                "error_type": type(e).__name__,
                "original_error": str(e),
                "stack_trace": traceback.format_exc(),
                "speaker_id": getattr(speaker, "id", None),
                "new_embedding_shape": getattr(new_embedding, "shape", None),
                "learning_rate": learning_rate
            }

            raise SpeakerIdentificationError(
                f"Failed to update speaker voice print: {str(e)}",
                {"error_type": "voice_print_update", **error_context}
            )

    def process_diarization_results(
        self,
        audio_path: str,
        diarization_results: Dict[str, Any],
        existing_speakers: List[Speaker]
    ) -> Dict[str, Any]:
        """Process diarization results to identify speakers.

        Args:
            audio_path: Path to the audio file
            diarization_results: Results from diarization
            existing_speakers: List of existing speakers to match against

        Returns:
            Dict[str, Any]: Processed results with identified speakers

        Raises:
            SpeakerIdentificationError: If processing fails
            FileNotFoundError: If the audio file does not exist
        """
        start_time = time.time()

        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Load audio file
            audio_data, sample_rate = self.file_handler.load_audio(audio_path)

            # Get segments from diarization results
            segments = diarization_results.get('segments', [])

            if not segments:
                logger.warning("No segments in diarization results")
                return diarization_results

            # Process each segment
            speaker_map = {}  # Map from diarization speaker ID to identified speaker
            segment_voice_prints = {}  # Cache voice prints by diarization speaker ID

            # Deep copy results to avoid modifying original
            processed_results = {
                'segments': [],
                'speakers': diarization_results.get('speakers', []),
                'identified_speakers': []
            }

            # First pass: generate voice prints for each unique speaker
            for segment in segments:
                # Check if processing time exceeds timeout
                if time.time() - start_time > self.timeout:
                    raise SpeakerIdentificationError(
                        f"Processing timeout after {self.timeout} seconds",
                        {
                            "error_type": "processing_timeout",
                            "timeout": self.timeout
                        }
                    )

                speaker_id = segment.get('speaker')

                # Skip if speaker already processed
                if speaker_id in segment_voice_prints:
                    continue

                # Extract audio segment
                start_time_sec = segment.get('start', 0)
                end_time_sec = segment.get('end', 0)

                if end_time_sec <= start_time_sec:
                    logger.warning(
                        f"Invalid segment time range: "
                        f"{start_time_sec}-{end_time_sec}"
                    )
                    continue

                # Skip very short segments
                if end_time_sec - start_time_sec < 1.0:
                    logger.debug(
                        "Segment too short for reliable speaker identification: "
                        f"{end_time_sec - start_time_sec}s"
                    )
                    continue

                try:
                    # Extract segment audio
                    segment_audio = self.file_handler.extract_segment(
                        audio_data, sample_rate, start_time_sec, end_time_sec
                    )

                    # Generate voice print
                    voice_print = self.generate_voice_print(
                        segment_audio, sample_rate
                    )

                    # Add to cache
                    segment_voice_prints[speaker_id] = voice_print

                except Exception as e:
                    logger.warning(
                        f"Failed to generate voice print for segment: {e}"
                    )
                    continue

            # Second pass: match speakers with existing speakers
            for speaker_id, voice_print in segment_voice_prints.items():
                if time.time() - start_time > self.timeout:
                    raise SpeakerIdentificationError(
                        f"Processing timeout after {self.timeout} seconds",
                        {
                            "error_type": "processing_timeout",
                            "timeout": self.timeout
                        }
                    )

                # Find matching speaker
                matching_speaker, similarity = self.find_matching_speaker(
                    voice_print, existing_speakers
                )

                if matching_speaker:
                    logger.info(
                        "Matched speaker {0} to existing speaker "
                        "{1} with similarity {2:.3f}".format(
                            speaker_id, matching_speaker.id, similarity
                        )
                    )
                    speaker_map[speaker_id] = matching_speaker

                    # Add to identified speakers if not already present
                    if matching_speaker.id not in [
                        s.get('id') for s in 
                        processed_results['identified_speakers']
                    ]:
                        processed_results['identified_speakers'].append({
                            'id': matching_speaker.id,
                            'name': matching_speaker.name,
                            'similarity': float(similarity)
                        })

            # Third pass: update segments with identified speakers
            for segment in segments:
                speaker_id = segment.get('speaker')
                segment_copy = segment.copy()

                # Add identified speaker if available
                if speaker_id in speaker_map:
                    identified_speaker = speaker_map[speaker_id]
                    segment_copy['identified_speaker'] = {
                        'id': identified_speaker.id,
                        'name': identified_speaker.name
                    }

                processed_results['segments'].append(segment_copy)

            return processed_results

        except (FileNotFoundError, ValueError):
            # Re-raise these as they are input validation errors
            raise

        except Exception as e:
            logger.error(f"Processing diarization results failed: {str(e)}")
            error_context = {
                "error_type": type(e).__name__,
                "original_error": str(e),
                "stack_trace": traceback.format_exc(),
                "audio_path": audio_path,
                "num_segments": len(diarization_results.get('segments', [])),
                "num_speakers": len(diarization_results.get('speakers', [])),
                "num_existing_speakers": len(existing_speakers),
                "processing_time": time.time() - start_time
            }

            raise SpeakerIdentificationError(
                f"Failed to process diarization results: {str(e)}",
                error_context
            )