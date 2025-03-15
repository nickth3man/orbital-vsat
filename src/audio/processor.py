"""
Audio processing pipeline for VSAT.

This module provides functionality for processing audio files through the complete pipeline.
"""

import os
import logging
import threading
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import numpy as np
import torch

from src.audio.file_handler import AudioFileHandler
from src.audio.audio_preprocessor import AudioPreprocessor
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.ml.diarization import Diarizer
from src.ml.speaker_identification import SpeakerIdentifier
from src.ml.voice_activity_detection import VoiceActivityDetector
from src.database.db_manager import DatabaseManager
from src.database.models import Recording, Speaker, TranscriptSegment, TranscriptWord
from src.utils.error_handler import ProcessingError, AudioError, FileError, ErrorSeverity

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Class for processing audio files through the complete pipeline."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                whisper_model_size: str = "medium",
                device: str = "cpu",
                hf_auth_token: Optional[str] = None,
                similarity_threshold: float = 0.75):
        """Initialize the audio processor.
        
        Args:
            db_manager: Database manager instance
            whisper_model_size: Size of the Whisper model to use
            device: Device to use for inference ("cpu" or "cuda")
            hf_auth_token: HuggingFace authentication token for pyannote.audio
            similarity_threshold: Threshold for speaker similarity (0-1)
        """
        logger.info("Initializing audio processor")
        
        # Initialize file handler
        self.file_handler = AudioFileHandler()
        
        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor()
        
        # Initialize voice activity detector
        self.voice_activity_detector = VoiceActivityDetector(
            auth_token=hf_auth_token,
            device=device
        )
        
        # Initialize database manager if not provided
        if db_manager is None:
            self.db_manager = DatabaseManager()
        else:
            self.db_manager = db_manager
        
        # Initialize transcriber and diarizer
        self.transcriber = WhisperTranscriber(model_size=whisper_model_size, device=device)
        self.diarizer = Diarizer(auth_token=hf_auth_token, device=device)
        
        # Initialize speaker identifier
        self.speaker_identifier = SpeakerIdentifier(
            auth_token=hf_auth_token, 
            device=device,
            similarity_threshold=similarity_threshold
        )
        
        logger.info("Audio processor initialized successfully")
    
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
            diarization = self.diarizer.diarize_file(file_path)
            
            # Update progress
            if progress_callback:
                progress_callback(0.5, "Performing transcription...")
            
            # Perform transcription
            transcription = self.transcriber.transcribe_file(file_path)
            
            # Update progress
            if progress_callback:
                progress_callback(0.7, "Performing speaker identification...")
            
            # Merge diarization and transcription
            merged_results = self.diarizer.merge_with_transcription(diarization, transcription)
            
            # Get all existing speakers from the database
            with self.db_manager.session_scope() as session:
                existing_speakers = session.query(Speaker).all()
                
                # Process diarization results to identify speakers
                identified_results = self.speaker_identifier.process_diarization_results(
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
                            updated_embedding = self.speaker_identifier.update_speaker_voice_print(
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
                        progress_callback: Optional[Callable[[float, str], None]] = None
                        ) -> List[str]:
        """Separate audio sources (speakers) using ML-based voice isolation.
        
        Args:
            audio_path: Path to audio file or audio data as numpy array
            sample_rate: Sample rate of the audio data (required if audio_path is array)
            max_speakers: Maximum number of speakers to separate
            model_type: Model type to use ('conv_tasnet' or 'sudo_rm_rf')
            output_dir: Directory to save separated audio files
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
                # Normalize
                source = source / np.max(np.abs(source)) * 0.9
                
                # Save to file
                output_file = os.path.join(output_dir, f"speaker_{i+1}.wav")
                AudioFileHandler.save_audio(output_file, source, sample_rate)
                output_files.append(output_file)
                
                # Report progress
                if progress_callback:
                    progress_percent = 0.7 + 0.3 * (i + 1) / actual_sources
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
            
            # Apply sensitivity preset
            self.voice_activity_detector.apply_sensitivity_preset(sensitivity_preset)
            
            # Detect speech segments
            segments = self.voice_activity_detector.detect_speech(
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
            
            # Load audio file
            audio_data, sample_rate = self.file_handler.load_file(file_path)
            
            # Detect speech segments
            segments = self.voice_activity_detector.detect_speech(file_path)
            
            # Calculate total duration
            total_duration = len(audio_data) / sample_rate
            
            # Calculate statistics
            stats = self.voice_activity_detector.calculate_speech_statistics(segments, total_duration)
            
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
        return self.voice_activity_detector.get_available_presets() 