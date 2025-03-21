"""
Audio processing pipeline for VSAT.

This module provides functionality for processing audio files through the complete pipeline.
"""

import os
import logging
import threading
import pickle
import tempfile
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import numpy as np
import torch
from pydub import AudioSegment

from src.audio.file_handler import AudioFileHandler
from src.audio.audio_preprocessor import AudioPreprocessor
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.ml.diarization import Diarizer
from src.ml.speaker_identification import SpeakerIdentifier
from src.ml.voice_activity_detection import VoiceActivityDetector
from src.database.db_manager import DatabaseManager
from src.database.models import Recording, Speaker, TranscriptSegment, TranscriptWord
from src.utils.error_handler import ProcessingError, AudioError, FileError, ErrorSeverity
from src.ml.model_manager import ModelManager, _model_lock
from src.ml.error_handling import AudioProcessingError, ModelLoadError, ResourceExhaustionError
from src.audio.chunked_processor import ChunkedProcessor
from src.config import Config

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Main audio processor class that integrates ML models for processing audio."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        whisper_model_size: str = "medium",
        device: str = "cpu",
        hf_auth_token: Optional[str] = None,
        similarity_threshold: float = 0.75,
        config: Optional[Config] = None,
        use_chunked_processing: bool = True,
        chunk_size_seconds: int = 300,  # 5 minutes
        chunk_overlap_seconds: int = 10,
        optimize_for_production: bool = True,
    ):
        """Initialize the audio processor.
        
        Args:
            db_manager: Database manager instance
            whisper_model_size: Size of the Whisper model to use
            device: Device to run the models on ('cpu' or 'cuda')
            hf_auth_token: Hugging Face authentication token
            similarity_threshold: Threshold for speaker similarity
            config: Configuration instance
            use_chunked_processing: Whether to process audio in chunks
            chunk_size_seconds: Size of chunks in seconds
            chunk_overlap_seconds: Overlap between chunks in seconds
            optimize_for_production: Whether to optimize models for production use
        """
        logger.info("Initializing audio processor")
        
        self.db_manager = db_manager
        self.config = config or Config()
        self.device = device
        self.whisper_model_size = whisper_model_size
        self.hf_auth_token = hf_auth_token
        self.similarity_threshold = similarity_threshold
        self.use_chunked_processing = use_chunked_processing
        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        
        # Initialize the model manager for centralized model management
        self.model_manager = ModelManager(
            device=device,
            hf_auth_token=hf_auth_token,
            optimize_for_production=optimize_for_production
        )
        
        # Initialize preprocessing tools
        self.preprocessor = AudioPreprocessor()
        
        # Initialize file handler
        self.file_handler = AudioFileHandler()
        
        # Models will be loaded lazily when needed through the _get methods
        
        logger.info("Audio processor initialized successfully")
    
    def _get_transcriber(self, device: Optional[str] = None) -> WhisperTranscriber:
        """Get a Whisper transcriber model from the model manager.
        
        Args:
            device: Optional device override
            
        Returns:
            WhisperTranscriber: Loaded transcriber model
        """
        try:
            return self.model_manager.get_whisper_transcriber(
                model_size=self.whisper_model_size,
                device=device or self.device
            )
        except (ModelLoadError, ResourceExhaustionError) as e:
            # Handle errors with proper context
            handle_errors(
                e,
                "Failed to load Whisper transcriber model",
                ErrorSeverity.HIGH
            )
            raise AudioProcessingError(
                "Transcription model initialization failed",
                "transcription",
                {"model_size": self.whisper_model_size, "device": self.device}
            )

    def _get_diarizer(self, device: Optional[str] = None) -> Diarizer:
        """Get a diarizer model from the model manager.
        
        Args:
            device: Optional device override
            
        Returns:
            Diarizer: Loaded diarizer model
        """
        try:
            return self.model_manager.get_diarizer(
                device=device or self.device
            )
        except (ModelLoadError, ResourceExhaustionError) as e:
            # Handle errors with proper context
            handle_errors(
                e,
                "Failed to load diarizer model",
                ErrorSeverity.HIGH
            )
            raise AudioProcessingError(
                "Diarization model initialization failed",
                "diarization",
                {"device": self.device}
            )

    def _get_speaker_identifier(self, device: Optional[str] = None) -> SpeakerIdentifier:
        """Get a speaker identifier model from the model manager.
        
        Args:
            device: Optional device override
            
        Returns:
            SpeakerIdentifier: Loaded speaker identifier model
        """
        try:
            return self.model_manager.get_speaker_identifier(
                device=device or self.device,
                similarity_threshold=self.similarity_threshold
            )
        except (ModelLoadError, ResourceExhaustionError) as e:
            # Handle errors with proper context
            handle_errors(
                e,
                "Failed to load speaker identifier model",
                ErrorSeverity.HIGH
            )
            raise AudioProcessingError(
                "Speaker identification model initialization failed",
                "speaker_identification",
                {"device": self.device, "threshold": self.similarity_threshold}
            )
            
    def _get_vad(self, device: Optional[str] = None) -> VoiceActivityDetector:
        """Get a voice activity detector from the model manager.
        
        Args:
            device: Optional device override
            
        Returns:
            VoiceActivityDetector: Loaded VAD model
        """
        try:
            return self.model_manager.get_voice_activity_detector(
                device=device or self.device
            )
        except (ModelLoadError, ResourceExhaustionError) as e:
            # Handle errors with proper context
            handle_errors(
                e,
                "Failed to load voice activity detection model",
                ErrorSeverity.MEDIUM
            )
            raise AudioProcessingError(
                "Voice activity detection model initialization failed",
                "vad",
                {"device": self.device}
            )

    def preprocess_audio(self, 
                        audio_data: np.ndarray, 
                        sample_rate: int,
                        preset: str = "default",
                        custom_settings: Optional[Dict[str, Any]] = None,
                        progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """Preprocess audio to improve quality.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            preset: Preset name from AudioPreprocessor.PRESETS or "custom"
            custom_settings: Custom settings to use if preset is "custom"
            progress_callback: Optional callback function for progress updates
            
        Returns:
            np.ndarray: Preprocessed audio data
            
        Raises:
            AudioError: If there's an error during preprocessing
        """
        return self.preprocessor.preprocess_audio(
            audio_data,
            sample_rate,
            preset=preset,
            custom_settings=custom_settings,
            progress_callback=progress_callback
        )
    
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
            preset: Preset name from AudioPreprocessor.PRESETS or "custom"
            custom_settings: Custom settings to use if preset is "custom"
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List[str]: List of paths to preprocessed files
            
        Raises:
            AudioError: If there's an error during preprocessing
        """
        return self.preprocessor.batch_preprocess(
            file_paths,
            output_dir,
            preset=preset,
            custom_settings=custom_settings,
            progress_callback=progress_callback
        )
    
    def get_available_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available preprocessing presets.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of preset names and their settings
        """
        return self.preprocessor.get_available_presets()
    
    def get_available_eq_profiles(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """Get available equalization profiles.
        
        Returns:
            Dict[str, List[Tuple[float, float, float]]]: Dictionary of profile names and their bands
        """
        return self.preprocessor.get_available_eq_profiles()
    
    def process_file(self, file_path: str, 
                    progress_callback: Optional[Callable[[float, str], None]] = None,
                    preprocess: bool = False,
                    preprocessing_preset: str = "default",
                    preprocessing_settings: Optional[Dict[str, Any]] = None) -> int:
        """Process an audio file through the complete pipeline.
        
        Args:
            file_path: Path to the audio file
            progress_callback: Optional callback function for progress updates
            preprocess: Whether to apply preprocessing before main processing
            preprocessing_preset: Preset name for preprocessing if enabled
            preprocessing_settings: Custom settings for preprocessing if preset is "custom"
            
        Returns:
            int: Recording ID in the database
            
        Raises:
            FileError: If there's an error with the file operations
            AudioError: If there's an error with audio processing
            ProcessingError: If there's an error with the processing pipeline
        """
        try:
            # Update progress
            if progress_callback:
                progress_callback(0.0, "Loading audio file...")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileError(
                    f"File not found: {file_path}",
                    ErrorSeverity.ERROR,
                    {"file_path": file_path}
                )
            
            # Check if file format is supported
            if not self.file_handler.is_supported_format(file_path):
                raise AudioError(
                    f"Unsupported audio format: {file_path}",
                    ErrorSeverity.ERROR,
                    {"file_path": file_path, "supported_formats": list(self.file_handler.SUPPORTED_FORMATS.keys())}
                )
            
            # Load audio file
            audio_data, sample_rate, metadata = self.file_handler.load_audio(file_path)
            
            # Apply preprocessing if enabled
            if preprocess:
                if progress_callback:
                    progress_callback(0.05, "Preprocessing audio...")
                
                # Define a preprocessing progress callback that scales to 5% of the overall progress
                def preprocess_progress(prog, status):
                    if progress_callback:
                        # Scale from 0.05 to 0.1
                        overall_prog = 0.05 + (prog * 0.05)
                        progress_callback(overall_prog, status)
                
                # Apply preprocessing
                audio_data = self.preprocess_audio(
                    audio_data,
                    sample_rate,
                    preset=preprocessing_preset,
                    custom_settings=preprocessing_settings,
                    progress_callback=preprocess_progress
                )
                
                # Update progress after preprocessing
                if progress_callback:
                    progress_callback(0.1, "Preprocessing complete")
            else:
                # Skip preprocessing progress portion
                if progress_callback:
                    progress_callback(0.1, "Starting transcription...")
            
            # Get audio info
            audio_info = self.file_handler.get_audio_info(file_path)
            
            # Create recording entry in database
            with self.db_manager.session_scope() as session:
                recording = Recording(
                    filename=os.path.basename(file_path),
                    path=os.path.abspath(file_path),
                    duration=audio_info['duration'],
                    sample_rate=audio_info['sample_rate'],
                    channels=audio_info['channels'],
                    processed=False
                )
                session.add(recording)
                session.flush()  # Get the ID without committing
                recording_id = recording.id
            
            # Update progress
            if progress_callback:
                progress_callback(0.3, "Performing speaker diarization...")
            
            # Perform diarization
            diarization = self._get_diarizer().diarize_file(file_path)
            
            # Update progress
            if progress_callback:
                progress_callback(0.5, "Performing transcription...")
            
            # Perform transcription
            transcription = self._get_transcriber().transcribe_file(file_path)
            
            # Update progress
            if progress_callback:
                progress_callback(0.7, "Performing speaker identification...")
            
            # Merge diarization and transcription
            merged_results = self._get_diarizer().merge_with_transcription(diarization, transcription)
            
            # Get all existing speakers from the database
            with self.db_manager.session_scope() as session:
                existing_speakers = session.query(Speaker).all()
                
                # Process diarization results to identify speakers
                identified_results = self._get_speaker_identifier().process_diarization_results(
                    file_path, merged_results, existing_speakers
                )
                
                # Update progress
                if progress_callback:
                    progress_callback(0.9, "Saving results to database...")
                
                # Process and store results
                self._process_results(session, recording_id, identified_results)
                
                # Mark recording as processed
                recording = session.query(Recording).get(recording_id)
                recording.processed = True
            
            # Update progress
            if progress_callback:
                progress_callback(1.0, "Processing complete")
            
            logger.info(f"Audio file processed successfully: {file_path}")
            return recording_id
            
        except (FileError, AudioError) as e:
            # These errors are already specific enough, so re-raise
            logger.error(f"Error processing audio file: {e}")
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.error(f"Unexpected error processing audio file: {e}")
            raise ProcessingError(
                f"Unexpected error during audio processing: {str(e)}",
                ErrorSeverity.ERROR, 
                {"file_path": file_path}
            ) from e
    
    def process_file_async(self, file_path: str, 
                          callback: Optional[Callable[[int, Optional[Exception]], None]] = None,
                          progress_callback: Optional[Callable[[float, str], None]] = None):
        """Process an audio file asynchronously.
        
        Args:
            file_path: Path to the audio file
            callback: Callback function called when processing is complete
            progress_callback: Callback function for progress updates
        """
        def _process_thread():
            try:
                recording_id = self.process_file(file_path, progress_callback)
                if callback:
                    callback(recording_id, None)
            except Exception as e:
                logger.error(f"Error in async processing: {e}")
                if callback:
                    callback(-1, e)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=_process_thread)
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started async processing of file: {file_path}")
    
    def _process_results(self, session, recording_id: int, results: Dict[str, Any]):
        """Process and store results in the database.
        
        Args:
            session: SQLAlchemy session
            recording_id: ID of the recording
            results: Merged results from diarization and transcription
        """
        try:
            logger.info(f"Processing results for recording {recording_id}")
            
            # Get the recording
            recording = session.query(Recording).get(recording_id)
            
            # Process speaker mapping
            speaker_mapping = {}
            if 'speaker_mapping' in results:
                for speaker_label, speaker_info in results['speaker_mapping'].items():
                    if speaker_info['is_new']:
                        # Create a new speaker
                        speaker = Speaker(
                            name=speaker_info['name'],
                            voice_print=pickle.dumps(speaker_info['embedding'])
                        )
                        session.add(speaker)
                        session.flush()  # Get the ID
                        speaker_mapping[speaker_label] = speaker
                    else:
                        # Get the existing speaker
                        speaker = session.query(Speaker).get(speaker_info['id'])
                        
                        # Update the speaker's voice print if it has changed
                        if 'embedding' in speaker_info:
                            updated_embedding = self._get_speaker_identifier().update_speaker_voice_print(
                                speaker, speaker_info['embedding']
                            )
                            speaker.voice_print = pickle.dumps(updated_embedding)
                        
                        speaker_mapping[speaker_label] = speaker
            
            # Process segments
            for segment in results['segments']:
                # Get or create speaker
                speaker = None
                if 'speaker' in segment:
                    speaker_label = segment['speaker']
                    
                    if speaker_label in speaker_mapping:
                        speaker = speaker_mapping[speaker_label]
                    else:
                        # Create a new speaker if not in mapping
                        speaker = Speaker(name=f"Speaker {speaker_label}")
                        session.add(speaker)
                        session.flush()  # Get the ID
                        speaker_mapping[speaker_label] = speaker
                
                # Create transcript segment
                transcript_segment = TranscriptSegment(
                    recording_id=recording_id,
                    speaker_id=speaker.id if speaker else None,
                    start_time=segment['start'],
                    end_time=segment['end'],
                    text=segment['text'] if 'text' in segment else None,
                    confidence=segment['confidence'] if 'confidence' in segment else None
                )
                session.add(transcript_segment)
                session.flush()  # Get the ID
                
                # Process words if available
                if 'words' in segment:
                    for word in segment['words']:
                        transcript_word = TranscriptWord(
                            segment_id=transcript_segment.id,
                            text=word['text'],
                            start_time=word['start'],
                            end_time=word['end'],
                            confidence=word['confidence'] if 'confidence' in word else None
                        )
                        session.add(transcript_word)
            
            logger.info(f"Results processed for recording {recording_id}")
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            session.rollback()
            raise 

    def validate_audio_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate an audio file before processing.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple containing:
                bool: True if valid, False otherwise
                Optional[str]: Error message if invalid, None otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            # Check if file format is supported
            if not self.file_handler.is_supported_format(file_path):
                return False, f"Unsupported audio format: {file_path}"
            
            # Try to load the file to check if it's valid
            try:
                self.file_handler.get_audio_info(file_path)
                return True, None
            except Exception as e:
                return False, f"Invalid audio file: {str(e)}"
            
        except Exception as e:
            return False, f"Error validating audio file: {str(e)}"

    def separate_sources(self, 
                        audio_path: Union[str, np.ndarray], 
                        sample_rate: Optional[int] = None,
                        max_speakers: int = 6,
                        model_type: str = "conv_tasnet",
                        output_dir: Optional[str] = None,
                        post_process: bool = True,
                        progress_callback: Optional[Callable[[float, str], None]] = None
                        ) -> List[str]:
        """Separate audio sources (speakers) using ML-based voice isolation.
        
        Args:
            audio_path: Path to audio file or audio data as numpy array
            sample_rate: Sample rate of the audio data (required if audio_path is array)
            max_speakers: Maximum number of speakers to separate
            model_type: Model type to use ('conv_tasnet', 'sudo_rm_rf', or 'kaituoxu')
            output_dir: Directory to save separated audio files
            post_process: Whether to apply post-processing to improve separation quality
            progress_callback: Function to call with progress updates
            
        Returns:
            List of paths to separated audio files
        
        Raises:
            ProcessingError: If an error occurs during processing
        """
        try:
            # Load audio if path provided
            if isinstance(audio_path, str):
                logger.info(f"Loading audio file for source separation: {audio_path}")
                audio_data, sample_rate, _ = AudioFileHandler.load_audio(audio_path)
                
                # Create output directory if needed
                if output_dir is None:
                    base_dir = os.path.dirname(audio_path)
                    filename = os.path.splitext(os.path.basename(audio_path))[0]
                    output_dir = os.path.join(base_dir, f"{filename}_separated")
                    
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            else:
                # Using provided audio data
                audio_data = audio_path
                if sample_rate is None:
                    raise ValueError("Sample rate must be provided when using audio data directly")
                
                # Create default output directory if needed
                if output_dir is None:
                    output_dir = os.path.join(os.getcwd(), "separated_sources")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
            
            # Report progress
            if progress_callback:
                progress_callback(0.1, "Loading separation model...")
            
            # Import models based on selected type
            if model_type == "conv_tasnet":
                from asteroid.models import ConvTasNet
                model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
                logger.info("Loaded ConvTasNet model")
            elif model_type == "sudo_rm_rf":
                from asteroid.models import SuDoRmRf
                model = SuDoRmRf.from_pretrained("julien-c/DPTNet-WHAM")
                logger.info("Loaded SuDoRmRf model")
            elif model_type == "kaituoxu":
                # Better performing model variant
                from asteroid.models import ConvTasNet
                model = ConvTasNet.from_pretrained("kaituoxu/Conv-TasNet-sepnoisy-16k")
                logger.info("Loaded kaituoxu ConvTasNet model (optimized for speech separation)")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Report progress
            if progress_callback:
                progress_callback(0.3, "Performing source separation...")
            
            # Perform separation
            with torch.no_grad():
                # Move model to appropriate device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                
                # Convert audio to tensor
                audio_tensor = torch.tensor(audio_data).unsqueeze(0).to(device)
                
                # Separate sources
                separated_sources = model(audio_tensor)
                
                # Convert back to numpy
                separated_sources = separated_sources.cpu().detach().numpy()
            
            # Limit to max_speakers
            actual_sources = min(separated_sources.shape[1], max_speakers)
            separated_sources = separated_sources[0, :actual_sources]
            
            # Report progress
            if progress_callback:
                progress_callback(0.7, f"Saving {actual_sources} separated sources...")
            
            # Save separated sources
            output_files = []
            for i, source in enumerate(separated_sources):
                # Apply post-processing if enabled
                if post_process:
                    if progress_callback:
                        progress_callback(0.7 + 0.1 * (i / actual_sources), 
                                         f"Post-processing source {i+1}/{actual_sources}")
                    
                    # Apply additional noise reduction to minimize bleed-through
                    source = self._post_process_separated_source(source, sample_rate)
                
                # Normalize
                source = source / np.max(np.abs(source)) * 0.9
                
                # Save to file
                output_file = os.path.join(output_dir, f"speaker_{i+1}.wav")
                AudioFileHandler.save_audio(output_file, source, sample_rate)
                output_files.append(output_file)
                
                # Report progress
                if progress_callback:
                    progress_percent = 0.8 + 0.2 * (i + 1) / actual_sources
                    progress_callback(progress_percent, 
                                     f"Saved source {i+1}/{actual_sources}")
            
            # Report completion
            if progress_callback:
                progress_callback(1.0, f"Completed separation of {len(output_files)} sources")
                
            logger.info(f"Completed source separation, saved {len(output_files)} files to {output_dir}")
            return output_files
            
        except Exception as e:
            error_msg = f"Error during source separation: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg, ErrorSeverity.ERROR, details={
                "audio_path": audio_path if isinstance(audio_path, str) else "numpy_array",
                "model_type": model_type,
                "error": str(e)
            }) 

    def detect_speech_segments(self, 
                              file_path: str, 
                              sensitivity_preset: str = "medium",
                              progress_callback: Optional[Callable[[float], None]] = None) -> List[Dict]:
        """Detect speech segments in an audio file.
        
        Args:
            file_path: Path to the audio file
            sensitivity_preset: Sensitivity preset to use ("high", "medium", "low", "very_low")
            progress_callback: Optional callback function for reporting progress (0-1)
            
        Returns:
            List[Dict]: List of speech segments with start/end times and confidence scores
            
        Raises:
            FileError: If the file cannot be loaded
            AudioError: If speech detection fails
        """
        try:
            logger.info(f"Detecting speech segments in {file_path}")
            
            # Get VAD model from model manager
            vad = self._get_vad()
            
            # Apply sensitivity preset
            vad.apply_sensitivity_preset(sensitivity_preset)
            
            # Detect speech segments
            segments = vad.detect_speech(
                file_path,
                progress_callback=progress_callback
            )
            
            logger.info(f"Detected {len(segments)} speech segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error detecting speech segments: {e}")
            if "No such file" in str(e):
                raise FileError(f"File not found: {file_path}", ErrorSeverity.ERROR)
            else:
                raise AudioError(f"Speech detection failed: {e}", ErrorSeverity.ERROR)
    
    def get_speech_statistics(self, file_path: str) -> Dict:
        """Calculate speech statistics for an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict: Statistics including speech percentage, count, avg duration, etc.
            
        Raises:
            FileError: If the file cannot be loaded
            AudioError: If statistics calculation fails
        """
        try:
            logger.info(f"Calculating speech statistics for {file_path}")
            
            # Get VAD model from model manager
            vad = self._get_vad()
            
            # Load audio file
            audio_data, sample_rate = self.file_handler.load_file(file_path)
            
            # Detect speech segments
            segments = vad.detect_speech(file_path)
            
            # Calculate total duration
            total_duration = len(audio_data) / sample_rate
            
            # Calculate statistics
            stats = vad.calculate_speech_statistics(segments, total_duration)
            
            logger.info(f"Speech statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating speech statistics: {e}")
            if "No such file" in str(e):
                raise FileError(f"File not found: {file_path}", ErrorSeverity.ERROR)
            else:
                raise AudioError(f"Statistics calculation failed: {e}", ErrorSeverity.ERROR)
    
    def get_vad_sensitivity_presets(self) -> List[str]:
        """Get list of available VAD sensitivity presets.
        
        Returns:
            List[str]: List of preset names
        """
        return self._get_vad().get_available_presets()

    def process_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Process an audio file for transcription, diarization, and speaker identification.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict: Processing results including transcription, diarization, and speakers
            
        Raises:
            AudioProcessingError: If processing fails
        """
        try:
            logger.info(f"Processing audio file: {file_path}")
            
            # Check if file exists and is accessible
            if not os.path.exists(file_path):
                raise AudioProcessingError(
                    f"Audio file does not exist: {file_path}",
                    "file_handling",
                    {"file_path": file_path}
                )
            
            # Preprocess audio file (convert to WAV, normalize, etc.)
            processed_file = self.preprocessor.preprocess(file_path)
            
            if self.use_chunked_processing:
                # Process in chunks for large files
                results = self._process_in_chunks(processed_file)
            else:
                # Process entire file at once
                results = self._process_full_file(processed_file)
            
            # Clean up temporary files
            if processed_file != file_path and os.path.exists(processed_file):
                os.remove(processed_file)
                
            return results
            
        except Exception as e:
            handle_errors(
                e,
                f"Error processing audio file: {file_path}",
                ErrorSeverity.HIGH
            )
            if isinstance(e, AudioProcessingError):
                raise
            else:
                raise AudioProcessingError(
                    f"Failed to process audio: {str(e)}",
                    "processing",
                    {"file_path": file_path, "error": str(e)}
                )
    
    def _process_in_chunks(self, file_path: str) -> Dict[str, Any]:
        """Process an audio file in chunks.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict: Combined processing results
        """
        logger.info(f"Processing audio in chunks: {file_path}")
        
        # Create chunked processor
        chunker = ChunkedProcessor(
            chunk_size_seconds=self.chunk_size_seconds,
            overlap_seconds=self.chunk_overlap_seconds,
            process_func=self._process_chunk
        )
        
        # Process audio in chunks
        return chunker.process_file(file_path)
    
    def _process_chunk(self, audio_segment: AudioSegment, chunk_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single audio chunk.
        
        Args:
            audio_segment: Audio segment to process
            chunk_info: Information about the chunk
            
        Returns:
            Dict: Processing results for the chunk
        """
        logger.debug(f"Processing chunk: {chunk_info}")
        
        # Save segment to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio_segment.export(temp_path, format="wav")
        
        try:
            # Process this chunk
            results = self._process_full_file(temp_path)
            
            # Add chunk information
            results["chunk_info"] = chunk_info
            
            return results
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _process_full_file(self, file_path: str) -> Dict[str, Any]:
        """Process a full audio file (or chunk).
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict: Processing results
        """
        # Get models from model manager
        transcriber = self._get_transcriber()
        diarizer = self._get_diarizer()
        speaker_identifier = self._get_speaker_identifier()
        
        # Run transcription
        transcription = transcriber.transcribe(file_path)
        
        # Run diarization
        diarization = diarizer.diarize(file_path)
        
        # Match transcription with diarization
        segments = self._align_transcription_and_diarization(transcription, diarization)
        
        # Run speaker identification on diarized segments
        speakers = speaker_identifier.identify_speakers(file_path, diarization)
        
        # Merge all results
        return {
            "transcription": transcription,
            "diarization": diarization,
            "segments": segments,
            "speakers": speakers
        }
    
    def _align_transcription_and_diarization(self, transcription: Dict[str, Any], 
                                            diarization: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Align transcription with diarization results.
        
        Args:
            transcription: Transcription results
            diarization: Diarization results
            
        Returns:
            List[Dict]: Aligned segments with speaker and text information
        """
        # Implementation of alignment algorithm
        # This is a simplified version that would need to be expanded
        aligned_segments = []
        
        # Extract transcription segments and diarization segments
        trans_segments = transcription.get("segments", [])
        diar_segments = diarization.get("segments", [])
        
        # Match segments based on time overlap
        for trans_seg in trans_segments:
            # Find overlapping diarization segments
            matching_speakers = []
            for diar_seg in diar_segments:
                # Check for time overlap
                if (trans_seg["start"] < diar_seg["end"] and 
                    trans_seg["end"] > diar_seg["start"]):
                    # Calculate overlap
                    overlap_start = max(trans_seg["start"], diar_seg["start"])
                    overlap_end = min(trans_seg["end"], diar_seg["end"])
                    overlap_duration = overlap_end - overlap_start
                    
                    # If significant overlap, assign this speaker
                    trans_duration = trans_seg["end"] - trans_seg["start"]
                    if overlap_duration / trans_duration > 0.5:
                        matching_speakers.append(diar_seg["speaker"])
            
            # Create aligned segment
            aligned_seg = {
                "start": trans_seg["start"],
                "end": trans_seg["end"],
                "text": trans_seg["text"],
                "speakers": matching_speakers
            }
            aligned_segments.append(aligned_seg)
        
        return aligned_segments
    
    def finish_batch_processing(self):
        """Finish batch processing and restore normal operation.
        
        Call this method after batch processing is complete to restore
        normal resource management behavior.
        """
        logger.info("Finishing batch processing")
        
        # Restore original cache timeout
        if hasattr(self, '_original_cache_timeout'):
            self.model_manager.cache_timeout = self._original_cache_timeout
            delattr(self, '_original_cache_timeout')
        
        # Perform cleanup to free resources
        self.model_manager._perform_cleanup()
        
    def cleanup(self):
        """Clean up resources when done."""
        logger.info("Cleaning up AudioProcessor resources")
        
        # Clean up model manager
        if hasattr(self, 'model_manager'):
            # Perform immediate cleanup of models
            with _model_lock:  # Use the model manager's lock if available
                model_keys = list(self.model_manager.models.keys()) if hasattr(self.model_manager, 'models') else []
                for model_key in model_keys:
                    try:
                        self.model_manager._unload_model(model_key)
                    except Exception as e:
                        logger.warning(f"Error unloading model {model_key}: {e}")
            
            # Shut down the model manager
            self.model_manager.shutdown()
        
        # Clean up other resources
        if hasattr(self, 'preprocessor') and hasattr(self.preprocessor, 'cleanup'):
            self.preprocessor.cleanup()
            
        if hasattr(self, 'file_handler') and hasattr(self.file_handler, 'cleanup'):
            self.file_handler.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {e}")
            
        logger.info("AudioProcessor resources cleaned up")

    def _post_process_separated_source(self, source: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply post-processing to a separated audio source to improve quality.
        
        This method applies several techniques to:
        1. Reduce bleed-through from other speakers
        2. Preserve voice characteristics 
        3. Improve overall separation quality
        
        Args:
            source: The separated audio source
            sample_rate: Sample rate of the audio
            
        Returns:
            np.ndarray: Processed audio source with improved quality
        """
        logger.debug("Applying post-processing to separated source")
        
        # Initialize preprocessor if needed
        if not hasattr(self, 'preprocessor'):
            from src.audio.audio_preprocessor import AudioPreprocessor
            self.preprocessor = AudioPreprocessor()
        
        # 1. Apply adaptive spectral gating noise reduction
        # Use energy-based threshold detection for optimal bleed-through removal
        try:
            # Calculate signal energy in speech frequency range (300-3400 Hz)
            from scipy import signal
            
            # Apply bandpass filter to focus on speech frequencies
            sos = signal.butter(10, [300, 3400], 'bandpass', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos, source)
            
            # Calculate energy in speech band
            speech_energy = np.mean(filtered**2)
            
            # Adaptive threshold based on speech energy
            adaptive_threshold = max(0.015, min(0.05, speech_energy * 0.5))
            adaptive_reduction = max(0.7, min(0.95, 0.65 + speech_energy * 2))
            
            logger.debug(f"Using adaptive threshold: {adaptive_threshold:.4f}, reduction: {adaptive_reduction:.4f}")
            
            # Apply noise reduction with adaptive parameters
            source = self.preprocessor.apply_noise_reduction(
                source, 
                sample_rate,
                threshold=adaptive_threshold,
                reduction_factor=adaptive_reduction
            )
        except Exception as e:
            logger.warning(f"Error in adaptive noise reduction: {e}")
            # Fallback to fixed parameters
            source = self.preprocessor.apply_noise_reduction(
                source, 
                sample_rate,
                threshold=0.025,
                reduction_factor=0.85
            )
        
        # 2. Apply voice-preserving spectral enhancement
        try:
            import librosa
            
            # Compute spectral contrast to enhance vocal formants
            S = librosa.stft(source)
            contrast_enhanced = np.abs(S) * 1.0
            
            # Enhance contrast in formant regions (500-3000 Hz)
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=2 * (S.shape[0] - 1))
            formant_mask = (freq_bins >= 500) & (freq_bins <= 3000)
            formant_indices = np.where(formant_mask)[0]
            
            if len(formant_indices) > 0:
                contrast_enhanced[formant_indices, :] *= 1.2
            
            # Convert back to time domain
            enhanced_source = librosa.istft(contrast_enhanced * np.exp(1j * np.angle(S)))
            
            # Normalize to original length
            if len(enhanced_source) > len(source):
                enhanced_source = enhanced_source[:len(source)]
            elif len(enhanced_source) < len(source):
                enhanced_source = np.pad(enhanced_source, (0, len(source) - len(enhanced_source)))
            
            # Mix with original to preserve transients
            source = enhanced_source * 0.7 + source * 0.3
        except Exception as e:
            logger.warning(f"Error in spectral enhancement: {e}")
            # Fallback to standard equalization
            source = self.preprocessor.apply_equalization(
                source,
                sample_rate,
                profile="speech_enhance"
            )
        
        # 3. Apply adaptive Wiener filtering with voice-focused parameters
        try:
            from scipy import signal
            
            # Estimate noise spectrum from low-energy frames
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            # Frame the signal
            n_frames = 1 + (len(source) - frame_length) // hop_length
            frames = np.zeros((n_frames, frame_length))
            for i in range(n_frames):
                frames[i] = source[i * hop_length:i * hop_length + frame_length]
            
            # Calculate frame energies
            frame_energies = np.sum(frames**2, axis=1)
            
            # Find low-energy frames (likely containing bleed-through/noise)
            energy_threshold = np.percentile(frame_energies, 15)
            noise_frames = frames[frame_energies < energy_threshold]
            
            # Estimate noise spectrum
            if len(noise_frames) > 0:
                noise_spectrum = np.mean(np.abs(np.fft.rfft(noise_frames, axis=1)), axis=0)
                # Apply Wiener filter with estimated noise
                wiener_filtered = np.zeros_like(source)
                for i in range(n_frames):
                    if i * hop_length + frame_length <= len(source):
                        frame = source[i * hop_length:i * hop_length + frame_length]
                        wiener_frame = signal.wiener(frame, mysize=frame_length, noise=noise_spectrum)
                        wiener_filtered[i * hop_length:i * hop_length + frame_length] += wiener_frame
                
                # Apply window overlap-add
                source = wiener_filtered
            else:
                # Fallback to standard Wiener filtering
                source = signal.wiener(source, mysize=512)
        except Exception as e:
            logger.warning(f"Error in adaptive Wiener filtering: {e}")
            try:
                # Simple fallback
                source = signal.wiener(source, mysize=512)
            except Exception:
                logger.warning("Wiener filtering unavailable")
        
        # 4. Apply harmonic-percussive source separation with voice optimization
        try:
            import librosa
            
            # Apply HP separation with parameters optimized for voice
            harmonic, percussive = librosa.effects.hpss(
                source, 
                kernel_size=31,  # Larger kernel for better voice separation
                power=2.0,       # Higher power for sharper separation
                mask=False       # Return audio not masks
            )
            
            # Enhance harmonic part in vocal frequency range
            S_harm = librosa.stft(harmonic)
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=2 * (S_harm.shape[0] - 1))
            
            # Create vocal enhancement mask (emphasize 300-3400 Hz)
            vocal_mask = np.ones(len(freq_bins))
            vocal_range = (freq_bins >= 300) & (freq_bins <= 3400)
            vocal_mask[vocal_range] = 1.3  # Boost vocal frequencies
            
            # Apply mask
            S_harm_enhanced = S_harm * vocal_mask[:, np.newaxis]
            harmonic_enhanced = librosa.istft(S_harm_enhanced)
            
            # Adjust to original length
            if len(harmonic_enhanced) > len(source):
                harmonic_enhanced = harmonic_enhanced[:len(source)]
            elif len(harmonic_enhanced) < len(source):
                harmonic_enhanced = np.pad(harmonic_enhanced, (0, len(source) - len(harmonic_enhanced)))
            
            # Mix enhanced harmonic with minimal percussive for naturalness
            source = harmonic_enhanced * 0.85 + percussive * 0.15
        except Exception as e:
            logger.warning(f"Error in harmonic-percussive separation: {e}")
            try:
                # Simpler fallback
                harmonic, percussive = librosa.effects.hpss(source)
                source = harmonic * 0.85 + percussive * 0.15
            except Exception:
                logger.warning("HPSS unavailable")
        
        # 5. Apply advanced multiband dynamic range compression 
        try:
            # Split into frequency bands for multiband processing
            from scipy.signal import butter, sosfilt
            
            # Define frequency bands relevant to speech
            bands = [
                (0, 250),       # Low frequencies
                (250, 1000),    # Low-mid (fundamental voice frequencies)
                (1000, 3500),   # Mid-high (voice formants)
                (3500, 8000),   # High (sibilance)
                (8000, 20000)   # Very high
            ]
            
            # Process each band separately
            band_signals = []
            
            for i, (low_freq, high_freq) in enumerate(bands):
                # Design bandpass filter
                if i == 0:
                    # Lowpass for first band
                    sos = butter(4, high_freq, 'lowpass', fs=sample_rate, output='sos')
                elif i == len(bands) - 1:
                    # Highpass for last band
                    sos = butter(4, low_freq, 'highpass', fs=sample_rate, output='sos')
                else:
                    # Bandpass for middle bands
                    sos = butter(4, [low_freq, high_freq], 'bandpass', fs=sample_rate, output='sos')
                
                # Apply filter
                band_signal = sosfilt(sos, source)
                
                # Apply band-specific compression
                # Stronger compression for mid bands where voice is most present
                if low_freq >= 250 and high_freq <= 3500:
                    # Voice bands get optimized compression
                    threshold = 0.07
                    ratio = 3.0
                    makeup_gain = 1.4
                else:
                    # Other bands get gentler compression
                    threshold = 0.12
                    ratio = 2.0
                    makeup_gain = 1.1
                
                # Apply compression
                amplitude = np.abs(band_signal)
                mask = amplitude > threshold
                compressed = np.copy(band_signal)
                compressed[mask] = threshold + (amplitude[mask] - threshold) / ratio
                compressed = compressed * np.sign(band_signal) * makeup_gain
                
                band_signals.append(compressed)
            
            # Mix all bands back together
            source = np.sum(band_signals, axis=0)
            
            # Blend with original to maintain naturalness
            source = source * 0.75 + source * 0.25
        except Exception as e:
            logger.warning(f"Error in multiband compression: {e}")
            try:
                # Fallback to simple full-band compression
                threshold = 0.1
                ratio = 3.0
                makeup_gain = 1.2
                
                amplitude = np.abs(source)
                mask = amplitude > threshold
                compressed = np.copy(source)
                compressed[mask] = threshold + (amplitude[mask] - threshold) / ratio
                compressed = compressed * np.sign(source) * makeup_gain
                
                source = compressed * 0.7 + source * 0.3
            except Exception:
                logger.warning("Compression processing failed")
        
        # 6. Apply de-essing to reduce sibilant bleed-through
        try:
            # Design high-shelf filter for sibilance range (4-8 kHz)
            from scipy.signal import butter, sosfilt
            
            # Extract sibilance band
            sos = butter(4, [4000, 8000], 'bandpass', fs=sample_rate, output='sos')
            sibilance = sosfilt(sos, source)
            
            # Apply dynamic compression only to sibilance
            thresh = 0.05
            ratio = 5.0
            
            # Apply compression to sibilance
            sib_amplitude = np.abs(sibilance)
            mask = sib_amplitude > thresh
            sibilance_compressed = np.copy(sibilance)
            sibilance_compressed[mask] = thresh + (sib_amplitude[mask] - thresh) / ratio
            
            # Remove the original sibilance and add back compressed sibilance
            sos_inverse = butter(4, [4000, 8000], 'bandstop', fs=sample_rate, output='sos')
            source_no_sib = sosfilt(sos_inverse, source)
            source = source_no_sib + sibilance_compressed * 0.7
        except Exception as e:
            logger.warning(f"Error in de-essing: {e}")
        
        # 7. Apply final adaptive limiting to prevent clipping while maintaining loudness
        try:
            # Calculate peak amplitude
            peak = np.max(np.abs(source))
            
            # Calculate appropriate gain to reach target without clipping
            target_peak = 0.9
            if peak > 0:
                gain = min(target_peak / peak, 2.0)  # Limit maximum gain
                source = source * gain
            
            # Apply soft limiting for extra protection
            limit_thresh = 0.8
            limit_samples = np.abs(source) > limit_thresh
            if np.any(limit_samples):
                source[limit_samples] = np.sign(source[limit_samples]) * (
                    limit_thresh + (np.abs(source[limit_samples]) - limit_thresh) / 3.0
                )
        except Exception as e:
            logger.warning(f"Error in final limiting: {e}")
            # Simple normalization as fallback
            source = source / (np.max(np.abs(source)) + 1e-8) * 0.9
        
        return source

    def preload_models(self, workflows: Optional[List[str]] = None):
        """Preload models based on anticipated workflows.
        
        This method preloads models that are likely to be needed based on
        specified workflows. This helps improve responsiveness when
        processing begins.
        
        Args:
            workflows: List of workflow names to preload models for.
                Supported workflows: 'transcription', 'diarization', 'speaker_id',
                'voice_activity', 'full_pipeline'
        """
        if workflows is None:
            # Default to preloading common models
            self.model_manager.preload_common_models()
            return
            
        models_to_preload = []
        
        for workflow in workflows:
            if workflow == 'transcription':
                models_to_preload.append({
                    'type': ModelManager.WHISPER,
                    'model_size': self.whisper_model_size,
                    'device': self.device
                })
            elif workflow == 'diarization':
                models_to_preload.append({
                    'type': ModelManager.DIARIZATION,
                    'device': self.device
                })
            elif workflow == 'speaker_id':
                models_to_preload.append({
                    'type': ModelManager.SPEAKER_ID,
                    'device': self.device,
                    'similarity_threshold': self.similarity_threshold
                })
            elif workflow == 'voice_activity':
                models_to_preload.append({
                    'type': ModelManager.VAD,
                    'device': self.device
                })
            elif workflow == 'full_pipeline':
                # Preload all models needed for full processing pipeline
                models_to_preload.extend([
                    {
                        'type': ModelManager.WHISPER,
                        'model_size': self.whisper_model_size,
                        'device': self.device
                    },
                    {
                        'type': ModelManager.DIARIZATION,
                        'device': self.device
                    },
                    {
                        'type': ModelManager.SPEAKER_ID,
                        'device': self.device,
                        'similarity_threshold': self.similarity_threshold
                    },
                    {
                        'type': ModelManager.VAD,
                        'device': self.device
                    }
                ])
        
        # Start preloading in background if we have models to preload
        if models_to_preload:
            logger.info(f"Preloading models for workflows: {', '.join(workflows)}")
            self.model_manager.preload_models(models_to_preload)
        else:
            logger.warning(f"No models to preload for workflows: {workflows}")
            
    def prepare_for_batch_processing(self):
        """Prepare the processor for batch processing multiple files.
        
        This method optimizes model loading and resource management for
        processing multiple files in sequence.
        """
        logger.info("Preparing for batch processing")
        
        # Preload all models needed for full processing pipeline
        self.preload_models(['full_pipeline'])
        
        # Configure model manager for batch processing
        # (longer cache timeout to avoid repeated loading)
        if hasattr(self.model_manager, 'cache_timeout'):
            # Save original timeout to restore later
            self._original_cache_timeout = self.model_manager.cache_timeout
            # Use longer timeout during batch processing
            self.model_manager.cache_timeout = 1800  # 30 minutes 