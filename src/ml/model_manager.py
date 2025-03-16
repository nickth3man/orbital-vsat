"""
ML Model Manager for VSAT.

This module provides a centralized management system for machine learning models,
optimizing resource usage, model loading, and inference performance.
"""

import os
import logging
import threading
import time
import torch
import gc
import psutil
import numpy as np
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from pathlib import Path
import traceback

from ..ml.diarization import Diarizer
from ..ml.speaker_identification import SpeakerIdentifier
from ..transcription.whisper_transcriber import WhisperTranscriber
from ..ml.voice_activity_detection import VoiceActivityDetector
from ..ml.voice_print_processor import VoicePrintProcessor
from ..ml.content_analysis import ContentAnalyzer
from ..utils.error_handler import ErrorSeverity
from ..ml.error_handling import ModelLoadError, ResourceExhaustionError

logger = logging.getLogger(__name__)

# Lock for thread-safe operations
_model_lock = threading.RLock()

class ModelInfo:
    """Information about a loaded model."""
    
    def __init__(self, model_instance: Any, model_type: str, device: str, 
                 last_used: float, size_mb: Optional[float] = None):
        """Initialize model info.
        
        Args:
            model_instance: The actual model instance
            model_type: Type of model (e.g., 'whisper', 'diarization')
            device: Device the model is loaded on ('cpu' or 'cuda')
            last_used: Timestamp of last use
            size_mb: Estimated memory size in MB
        """
        self.model_instance = model_instance
        self.model_type = model_type
        self.device = device
        self.last_used = last_used
        self.size_mb = size_mb
        self.is_loading = False
        self.loading_thread = None
        self.ready_event = threading.Event()

class ModelManager:
    """Central manager for ML models used in VSAT."""
    
    # Default locations for model storage
    DEFAULT_MODEL_DIR = str(Path.home() / '.vsat' / 'models')
    
    # Model type constants
    WHISPER = "whisper"
    DIARIZATION = "diarization"
    SPEAKER_ID = "speaker_id"
    VAD = "vad"
    VOICE_PRINT = "voice_print"
    CONTENT_ANALYSIS = "content_analysis"
    
    # Maximum memory usage thresholds (percentages)
    MAX_RAM_USAGE = 75.0
    MAX_GPU_MEM_USAGE = 85.0
    
    # Cache timeout in seconds
    DEFAULT_CACHE_TIMEOUT = 300  # 5 minutes
    
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 cache_timeout: float = DEFAULT_CACHE_TIMEOUT,
                 device: str = "cpu",
                 hf_auth_token: Optional[str] = None,
                 optimize_for_production: bool = True):
        """Initialize the model manager.
        
        Args:
            model_dir: Directory to store models
            cache_timeout: Time in seconds to keep unused models in memory
            device: Default device to load models on ('cpu' or 'cuda')
            hf_auth_token: HuggingFace authentication token
            optimize_for_production: Whether to optimize models for production
        """
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.cache_timeout = cache_timeout
        self.default_device = self._validate_device(device)
        self.hf_auth_token = hf_auth_token
        self.optimize_for_production = optimize_for_production
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model cache dictionary
        self.models: Dict[str, ModelInfo] = {}
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.stop_cleanup = threading.Event()

        # Start the cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"Model manager initialized (device: {self.default_device}, "
                   f"cache_timeout: {self.cache_timeout}s, "
                   f"optimize_for_production: {self.optimize_for_production})")
    
    def _validate_device(self, device: str) -> str:
        """Validate the specified device.
        
        Args:
            device: Device to validate ('cpu' or 'cuda')
            
        Returns:
            str: Validated device ('cpu' or 'cuda')
        """
        if device.lower() == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return device.lower()
    
    def _start_cleanup_thread(self):
        """Start the background thread that cleans up unused models."""
        if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
            return
        
        self.stop_cleanup.clear()
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_task,
            name="ModelCleanupThread",
            daemon=True
        )
        self.cleanup_thread.start()
        logger.debug("Started model cleanup thread")
    
    def _cleanup_task(self):
        """Background task to clean up unused models."""
        while not self.stop_cleanup.is_set():
            try:
                self._perform_cleanup()
            except Exception as e:
                logger.error(f"Error in model cleanup task: {e}")
            
            # Sleep for a while before checking again
            self.stop_cleanup.wait(60)  # Check every minute
    
    def _perform_cleanup(self):
        """Perform cleanup of unused models."""
        current_time = time.time()
        
        with _model_lock:
            # Find models that haven't been used for a while
            expired_models = []
            for model_key, model_info in self.models.items():
                if (current_time - model_info.last_used > self.cache_timeout and
                    not model_info.is_loading):
                    expired_models.append(model_key)
            
            # Unload expired models
            for model_key in expired_models:
                self._unload_model(model_key)
    
    def _unload_model(self, model_key: str):
        """Unload a model from memory.
        
        Args:
            model_key: Key of the model to unload
        """
        if model_key not in self.models:
            return
        
        model_info = self.models[model_key]
        logger.info(f"Unloading unused model: {model_key} (type: {model_info.model_type}, "
                   f"device: {model_info.device})")
        
        # Proper cleanup based on model type and device
        model_instance = model_info.model_instance
        
        try:
            # Remove model from dictionary
            del self.models[model_key]
            
            # CPU models just need reference removal for GC
            if model_info.device == "cuda":
                # For PyTorch models, we need special handling
                if hasattr(model_instance, "to"):
                    model_instance.to("cpu")
                
                # Manual GPU memory cleanup
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug(f"Successfully unloaded model: {model_key}")
        except Exception as e:
            logger.error(f"Error unloading model {model_key}: {e}")
    
    def _check_resources(self, device: str, required_memory_mb: Optional[float] = None) -> bool:
        """Check if there are enough resources to load a model.
        
        Args:
            device: Device to check ('cpu' or 'cuda')
            required_memory_mb: Estimated memory requirement in MB
            
        Returns:
            bool: True if there are enough resources, False otherwise
        """
        try:
            if device == "cuda":
                # Check GPU memory
                if not torch.cuda.is_available():
                    return False
                
                total_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = total_memory - torch.cuda.memory_reserved(0)
                free_percent = (free_memory / total_memory) * 100
                
                if required_memory_mb and free_memory < required_memory_mb * 1024 * 1024:
                    return False
                
                return free_percent >= (100 - self.MAX_GPU_MEM_USAGE)
            else:
                # Check system memory
                mem_info = psutil.virtual_memory()
                free_percent = mem_info.available / mem_info.total * 100
                
                if required_memory_mb and mem_info.available < required_memory_mb * 1024 * 1024:
                    return False
                
                return free_percent >= (100 - self.MAX_RAM_USAGE)
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return True  # Assume there are enough resources if check fails
    
    def _get_model_dir(self, model_type: str) -> str:
        """Get the directory for a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            str: Path to model directory
        """
        model_subdir = {
            self.WHISPER: "whisper",
            self.DIARIZATION: "diarization",
            self.SPEAKER_ID: "speaker_id",
            self.VAD: "vad",
            self.VOICE_PRINT: "voice_print",
            self.CONTENT_ANALYSIS: "content_analysis"
        }.get(model_type, model_type)
        
        model_dir = os.path.join(self.model_dir, model_subdir)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def _get_model_key(self, model_type: str, **params) -> str:
        """Generate a unique key for a model.
        
        Args:
            model_type: Type of model
            **params: Model parameters
            
        Returns:
            str: Unique model key
        """
        # Sort parameters for consistency
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()) 
                           if k not in ["auth_token", "hf_auth_token"])
        return f"{model_type}_{param_str}"
    
    def get_whisper_transcriber(self, 
                               model_size: str = "medium", 
                               device: Optional[str] = None,
                               compute_type: str = "float32",
                               use_word_aligner: bool = True,
                               async_load: bool = False) -> WhisperTranscriber:
        """Get or load a Whisper transcriber model.
        
        Args:
            model_size: Size of the Whisper model
            device: Device to load the model on
            compute_type: Compute type for inference
            use_word_aligner: Whether to use word aligner
            async_load: Whether to load the model asynchronously
            
        Returns:
            WhisperTranscriber: The requested model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        device = self._validate_device(device or self.default_device)
        
        # Choose appropriate compute_type for production
        if self.optimize_for_production and device == "cuda":
            # Use mixed precision for CUDA
            compute_type = "float16"
        
        model_key = self._get_model_key(
            self.WHISPER, 
            size=model_size, 
            device=device, 
            compute_type=compute_type,
            use_word_aligner=use_word_aligner
        )
        
        # Estimated model sizes in MB based on model size
        model_size_estimates = {
            "tiny": 150,
            "base": 300,
            "small": 500,
            "medium": 1500,
            "large-v1": 3000,
            "large-v2": 3000,
            "large-v3": 3000
        }
        
        required_memory = model_size_estimates.get(model_size, 1500)
        model_dir = self._get_model_dir(self.WHISPER)
        
        return self._get_or_load_model(
            model_key,
            model_type=self.WHISPER,
            load_func=lambda: WhisperTranscriber(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                download_root=model_dir,
                use_word_aligner=use_word_aligner,
                timeout=600  # Longer timeout for model loading
            ),
            device=device,
            required_memory_mb=required_memory,
            async_load=async_load
        )
    
    def get_diarizer(self,
                    device: Optional[str] = None,
                    async_load: bool = False) -> Diarizer:
        """Get or load a speaker diarization model.
        
        Args:
            device: Device to load the model on
            async_load: Whether to load the model asynchronously
            
        Returns:
            Diarizer: The requested model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        device = self._validate_device(device or self.default_device)
        model_key = self._get_model_key(self.DIARIZATION, device=device)
        model_dir = self._get_model_dir(self.DIARIZATION)
        
        return self._get_or_load_model(
            model_key,
            model_type=self.DIARIZATION,
            load_func=lambda: Diarizer(
                auth_token=self.hf_auth_token,
                device=device,
                download_root=model_dir,
                timeout=600
            ),
            device=device,
            required_memory_mb=1000,  # Approximate memory requirement
            async_load=async_load
        )
    
    def get_speaker_identifier(self, 
                              device: Optional[str] = None,
                              similarity_threshold: float = 0.75,
                              async_load: bool = False) -> SpeakerIdentifier:
        """Get or load a speaker identification model.
        
        Args:
            device: Device to load the model on
            similarity_threshold: Threshold for speaker similarity
            async_load: Whether to load the model asynchronously
            
        Returns:
            SpeakerIdentifier: The requested model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        device = self._validate_device(device or self.default_device)
        model_key = self._get_model_key(
            self.SPEAKER_ID, 
            device=device, 
            threshold=similarity_threshold
        )
        model_dir = self._get_model_dir(self.SPEAKER_ID)
        
        return self._get_or_load_model(
            model_key,
            model_type=self.SPEAKER_ID,
            load_func=lambda: SpeakerIdentifier(
                auth_token=self.hf_auth_token,
                device=device,
                download_root=model_dir,
                similarity_threshold=similarity_threshold,
                timeout=600
            ),
            device=device,
            required_memory_mb=500,  # Approximate memory requirement
            async_load=async_load
        )
    
    def get_voice_activity_detector(self,
                                  device: Optional[str] = None,
                                  async_load: bool = False) -> VoiceActivityDetector:
        """Get or load a voice activity detection model.
        
        Args:
            device: Device to load the model on
            async_load: Whether to load the model asynchronously
            
        Returns:
            VoiceActivityDetector: The requested model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        device = self._validate_device(device or self.default_device)
        model_key = self._get_model_key(self.VAD, device=device)
        model_dir = self._get_model_dir(self.VAD)
        
        return self._get_or_load_model(
            model_key,
            model_type=self.VAD,
            load_func=lambda: VoiceActivityDetector(
                auth_token=self.hf_auth_token,
                device=device,
                download_root=model_dir
            ),
            device=device,
            required_memory_mb=300,  # Approximate memory requirement
            async_load=async_load
        )
    
    def get_content_analyzer(self,
                           device: Optional[str] = None,
                           async_load: bool = False) -> ContentAnalyzer:
        """Get or load a content analysis model.
        
        Args:
            device: Device to load the model on
            async_load: Whether to load the model asynchronously
            
        Returns:
            ContentAnalyzer: The requested model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        device = self._validate_device(device or self.default_device)
        model_key = self._get_model_key(self.CONTENT_ANALYSIS, device=device)
        model_dir = self._get_model_dir(self.CONTENT_ANALYSIS)
        
        return self._get_or_load_model(
            model_key,
            model_type=self.CONTENT_ANALYSIS,
            load_func=lambda: ContentAnalyzer(
                device=device,
                download_root=model_dir
            ),
            device=device,
            required_memory_mb=500,  # Approximate memory requirement
            async_load=async_load
        )
    
    def _check_resource_pressure(self) -> float:
        """Check the current resource pressure on the system.
        
        Returns:
            float: A value between 0 and 1 indicating resource pressure (higher = more pressure)
        """
        try:
            if self.default_device == "cuda" and torch.cuda.is_available():
                # Calculate GPU memory pressure
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                cached_memory = torch.cuda.memory_reserved(0) - allocated_memory
                
                # Calculate pressure (0-1)
                gpu_pressure = allocated_memory / total_memory
                
                # Get CPU pressure as well
                mem_info = psutil.virtual_memory()
                cpu_pressure = 1.0 - (mem_info.available / mem_info.total)
                
                # Return higher of the two pressures
                return max(gpu_pressure, cpu_pressure)
            else:
                # Only check CPU memory pressure
                mem_info = psutil.virtual_memory()
                return 1.0 - (mem_info.available / mem_info.total)
        except Exception as e:
            logger.warning(f"Error checking resource pressure: {e}")
            return 0.5  # Default to medium pressure on error
    
    def _perform_memory_cleanup_if_needed(self, required_memory_mb: Optional[float] = None) -> bool:
        """Perform memory cleanup if system is under pressure.
        
        Args:
            required_memory_mb: Estimated memory required for upcoming operation
            
        Returns:
            bool: True if cleanup was performed, False otherwise
        """
        # Check current resource pressure
        pressure = self._check_resource_pressure()
        
        # Cleanup thresholds
        light_cleanup_threshold = 0.7  # 70% resource usage
        heavy_cleanup_threshold = 0.85  # 85% resource usage
        
        # If we need specific memory and can't satisfy it, force cleanup
        force_cleanup = False
        if required_memory_mb is not None:
            if self.default_device == "cuda" and torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                force_cleanup = (free_memory < required_memory_mb * 1024 * 1024)
            else:
                mem_info = psutil.virtual_memory()
                force_cleanup = (mem_info.available < required_memory_mb * 1024 * 1024)
        
        if pressure > heavy_cleanup_threshold or force_cleanup:
            # Heavy cleanup - unload all but the most recently used models
            logger.info(f"System under heavy resource pressure ({pressure:.2f}), performing thorough cleanup")
            return self._unload_models_by_priority(max_to_keep=2)
            
        elif pressure > light_cleanup_threshold:
            # Light cleanup - only unload models that haven't been used in a while
            logger.info(f"System under moderate resource pressure ({pressure:.2f}), performing light cleanup")
            return self._unload_models_by_priority(max_to_keep=4)
            
        return False
    
    def _unload_models_by_priority(self, max_to_keep: int = 2) -> bool:
        """Unload models by priority (least recently used first).
        
        Args:
            max_to_keep: Maximum number of models to keep
            
        Returns:
            bool: True if any models were unloaded, False otherwise
        """
        with _model_lock:
            if len(self.models) <= max_to_keep:
                return False
            
            # Sort models by last used time (oldest first)
            sorted_models = sorted(
                [(key, info) for key, info in self.models.items() if not info.is_loading],
                key=lambda x: x[1].last_used
            )
            
            # Keep the most recently used models
            models_to_unload = sorted_models[:-max_to_keep]
            
            # Unload selected models
            for model_key, _ in models_to_unload:
                self._unload_model(model_key)
            
            return len(models_to_unload) > 0
    
    def _get_or_load_model(self, 
                          model_key: str, 
                          model_type: str,
                          load_func: Callable[[], Any],
                          device: str,
                          required_memory_mb: Optional[float] = None,
                          async_load: bool = False) -> Any:
        """Get a model from cache or load it.
        
        Args:
            model_key: Unique model key
            model_type: Type of model
            load_func: Function to load the model
            device: Device to load the model on
            required_memory_mb: Estimated memory requirement in MB
            async_load: Whether to load the model asynchronously
            
        Returns:
            Any: The requested model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        with _model_lock:
            # Check if model is already in cache
            if model_key in self.models:
                model_info = self.models[model_key]
                
                # If model is still loading, wait for it
                if model_info.is_loading:
                    logger.debug(f"Model {model_key} is currently loading, waiting...")
                    if async_load:
                        # Return a loading handle for async loading
                        return model_info
                    
                    # Wait for loading to complete
                    ready = model_info.ready_event.wait(timeout=300)
                    if not ready:
                        raise ModelLoadError(
                            f"Timed out waiting for model {model_key} to load",
                            model_type,
                            {"device": device, "timeout": 300}
                        )
                
                # Update last used timestamp
                model_info.last_used = time.time()
                return model_info.model_instance
            
            # Perform cleanup if system is under resource pressure
            self._perform_memory_cleanup_if_needed(required_memory_mb)
            
            # Check if there are enough resources after cleanup
            if not self._check_resources(device, required_memory_mb):
                logger.warning(f"Insufficient resources for model {model_key} on {device}")
                
                if device != "cpu":
                    # Fall back to CPU
                    logger.info(f"Falling back to CPU for model {model_key}")
                    return self._get_or_load_model(
                        model_key.replace(f"device={device}", "device=cpu"),
                        model_type,
                        lambda: load_func(),  # Will be updated inside function to use CPU
                        "cpu",
                        required_memory_mb,
                        async_load
                    )
                else:
                    raise ResourceExhaustionError(
                        f"Insufficient resources to load model {model_key}",
                        "memory",
                        {"device": device, "required_memory_mb": required_memory_mb}
                    )
            
            # Create model info placeholder
            model_info = ModelInfo(
                model_instance=None,
                model_type=model_type,
                device=device,
                last_used=time.time(),
                size_mb=required_memory_mb
            )
            model_info.is_loading = True
            model_info.ready_event.clear()
            
            # Add to cache
            self.models[model_key] = model_info
            
            # Define loading function
            def _load_model():
                try:
                    logger.info(f"Loading model: {model_key} on {device}")
                    model_instance = load_func()
                    
                    with _model_lock:
                        if model_key in self.models:
                            # Update model info
                            model_info = self.models[model_key]
                            model_info.model_instance = model_instance
                            model_info.is_loading = False
                            model_info.last_used = time.time()
                            model_info.ready_event.set()
                            
                            logger.info(f"Successfully loaded model: {model_key}")
                            return model_instance
                        else:
                            # Model was removed during loading
                            logger.warning(f"Model {model_key} was removed during loading")
                            return None
                except Exception as e:
                    logger.error(f"Error loading model {model_key}: {e}")
                    logger.error(traceback.format_exc())
                    
                    with _model_lock:
                        if model_key in self.models:
                            # Clean up failed model
                            del self.models[model_key]
                    
                    # Re-raise as ModelLoadError
                    raise ModelLoadError(
                        f"Failed to load model {model_key}: {str(e)}",
                        model_type,
                        {"device": device, "error": str(e), "traceback": traceback.format_exc()}
                    )
            
            # Load model asynchronously or synchronously
            if async_load:
                # Start loading thread
                model_info.loading_thread = threading.Thread(
                    target=_load_model,
                    name=f"ModelLoader-{model_key}",
                    daemon=True
                )
                model_info.loading_thread.start()
                return model_info
            else:
                # Load synchronously
                return _load_model()
    
    def wait_for_model(self, model_info: ModelInfo, timeout: float = 300) -> Any:
        """Wait for an asynchronously loading model to complete.
        
        Args:
            model_info: Model info from async load
            timeout: Maximum time to wait in seconds
            
        Returns:
            Any: The loaded model
            
        Raises:
            ModelLoadError: If loading times out or fails
        """
        if not model_info.is_loading:
            return model_info.model_instance
        
        logger.debug(f"Waiting for model {model_info.model_type} to load...")
        ready = model_info.ready_event.wait(timeout=timeout)
        
        if not ready:
            raise ModelLoadError(
                f"Timed out waiting for model to load",
                model_info.model_type,
                {"device": model_info.device, "timeout": timeout}
            )
        
        return model_info.model_instance
    
    def unload_all_models(self):
        """Unload all models from memory."""
        with _model_lock:
            model_keys = list(self.models.keys())
            for model_key in model_keys:
                self._unload_model(model_key)
    
    def shutdown(self):
        """Shutdown the model manager and clean up resources."""
        logger.info("Shutting down model manager")
        
        # Stop cleanup thread
        self.stop_cleanup.set()
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        # Unload all models
        self.unload_all_models()
        
        logger.info("Model manager shutdown complete")

    def preload_models(self, models_to_preload: List[Dict[str, Any]], timeout: Optional[float] = None):
        """Preload models in the background during idle time.
        
        This method helps improve responsiveness by loading models in advance
        when they are likely to be needed.
        
        Args:
            models_to_preload: List of model specifications to preload.
                Each item should be a dictionary with keys:
                - 'type': Model type (e.g., 'whisper', 'diarization')
                - Other parameters specific to the model type
            timeout: Optional timeout in seconds for preloading
        """
        # Check if we have enough resources to preload
        pressure = self._check_resource_pressure()
        if pressure > 0.6:  # Only preload if system has resources available
            logger.info(f"System resource pressure too high ({pressure:.2f}), skipping preload")
            return
            
        logger.info(f"Preloading {len(models_to_preload)} models in background")
        
        # Start background threads for each model
        loading_threads = []
        for model_spec in models_to_preload:
            model_type = model_spec.get('type')
            if not model_type:
                logger.warning(f"Skipping preload of model with missing type: {model_spec}")
                continue
                
            # Create loading thread based on model type
            if model_type == self.WHISPER:
                thread = threading.Thread(
                    target=self.get_whisper_transcriber,
                    kwargs={
                        'model_size': model_spec.get('model_size', 'medium'),
                        'device': model_spec.get('device', self.default_device),
                        'async_load': True
                    },
                    daemon=True
                )
            elif model_type == self.DIARIZATION:
                thread = threading.Thread(
                    target=self.get_diarizer,
                    kwargs={
                        'device': model_spec.get('device', self.default_device),
                        'async_load': True
                    },
                    daemon=True
                )
            elif model_type == self.SPEAKER_ID:
                thread = threading.Thread(
                    target=self.get_speaker_identifier,
                    kwargs={
                        'device': model_spec.get('device', self.default_device),
                        'similarity_threshold': model_spec.get('similarity_threshold', 0.75),
                        'async_load': True
                    },
                    daemon=True
                )
            elif model_type == self.VAD:
                thread = threading.Thread(
                    target=self.get_voice_activity_detector,
                    kwargs={
                        'device': model_spec.get('device', self.default_device),
                        'async_load': True
                    },
                    daemon=True
                )
            else:
                logger.warning(f"Unknown model type for preloading: {model_type}")
                continue
                
            # Start loading thread
            thread.name = f"Preloader-{model_type}"
            thread.start()
            loading_threads.append(thread)
            
        # Wait for threads to complete if timeout specified
        if timeout is not None and loading_threads:
            end_time = time.time() + timeout
            for thread in loading_threads:
                wait_time = max(0, end_time - time.time())
                if wait_time > 0:
                    thread.join(timeout=wait_time)
                    
        logger.debug(f"Preload initiated for {len(loading_threads)} models")

    def preload_common_models(self):
        """Preload commonly used models for quicker access later.
        
        This method preloads the most commonly needed models during
        system initialization or idle time.
        """
        # Determine which models to preload based on available resources
        preload_specs = []
        
        # On systems with sufficient resources, preload more models
        if self._check_resource_pressure() < 0.4:  # Low resource pressure
            # Preload on the default device
            preload_specs = [
                {'type': self.WHISPER, 'model_size': 'medium'},
                {'type': self.DIARIZATION},
                {'type': self.VAD}
            ]
            
            # If GPU is available with low usage, preload there too
            if self.default_device == "cpu" and torch.cuda.is_available():
                gpu_pressure = self._get_gpu_memory_pressure()
                if gpu_pressure < 0.3:  # Low GPU usage
                    preload_specs.append({'type': self.WHISPER, 'model_size': 'medium', 'device': 'cuda'})
        else:
            # Just preload the VAD model which is small and frequently used
            preload_specs = [{'type': self.VAD}]
        
        # Start preloading in background
        self.preload_models(preload_specs)
        
    def _get_gpu_memory_pressure(self) -> float:
        """Get GPU memory pressure.
        
        Returns:
            float: GPU memory pressure (0-1) or -1 if GPU is not available
        """
        if not torch.cuda.is_available():
            return -1
            
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            return allocated_memory / total_memory
        except Exception as e:
            logger.warning(f"Error getting GPU memory pressure: {e}")
            return -1 