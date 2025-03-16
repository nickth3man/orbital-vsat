#!/usr/bin/env python3
"""
Batch processor for audio files.

This module provides functionality for processing multiple audio files
in batch mode with optimized resource usage and parallel processing.
"""

import os
import logging
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

from src.audio.file_handler import AudioFileHandler
from src.audio.processor import AudioProcessor
from src.database.data_manager import DataManager
from src.utils.performance_optimizer import (
    get_performance_monitor,
    get_resource_optimizer,
    get_batch_optimizer,
    timed
)
from src.utils.error_handler import VSATError, ProcessingError

logger = logging.getLogger(__name__)

class BatchProcessingTask:
    """Represents a single file processing task in a batch."""
    
    def __init__(self, file_path: str, options: Dict[str, Any] = None):
        """Initialize a batch processing task.
        
        Args:
            file_path: Path to the audio file
            options: Processing options
        """
        self.file_path = file_path
        self.options = options or {}
        self.file_size_mb = self._get_file_size_mb()
        self.status = "pending"  # pending, processing, completed, failed
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.processing_time = None
    
    def _get_file_size_mb(self) -> float:
        """Get the file size in megabytes.
        
        Returns:
            File size in megabytes
        """
        try:
            return os.path.getsize(self.file_path) / (1024 * 1024)
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Could not get file size for {self.file_path}: {str(e)}")
            return 0.0
    
    def mark_started(self):
        """Mark the task as started."""
        self.status = "processing"
        self.start_time = time.time()
    
    def mark_completed(self, result: Any):
        """Mark the task as completed.
        
        Args:
            result: Processing result
        """
        self.status = "completed"
        self.result = result
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
    
    def mark_failed(self, error: Exception):
        """Mark the task as failed.
        
        Args:
            error: Exception that caused the failure
        """
        self.status = "failed"
        self.error = error
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
    
    def __str__(self) -> str:
        """Get a string representation of the task.
        
        Returns:
            String representation
        """
        return f"BatchProcessingTask({os.path.basename(self.file_path)}, {self.status}, {self.file_size_mb:.2f}MB)"


class BatchProcessor:
    """Process multiple audio files in batch mode with optimized resource usage."""
    
    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        max_concurrent_tasks: Optional[int] = None,
        enable_performance_monitoring: bool = True
    ):
        """Initialize the batch processor.
        
        Args:
            data_manager: Data manager for storing results
            max_concurrent_tasks: Maximum number of concurrent tasks (None for auto)
            enable_performance_monitoring: Whether to enable performance monitoring
        """
        self.data_manager = data_manager
        self.file_handler = AudioFileHandler()
        
        # Performance optimization
        self.performance_monitor = get_performance_monitor()
        self.resource_optimizer = get_resource_optimizer()
        self.batch_optimizer = get_batch_optimizer()
        
        # Set max concurrent tasks (auto-determine if not specified)
        if max_concurrent_tasks is None:
            self.max_concurrent_tasks = self.batch_optimizer.get_max_concurrent_tasks()
        else:
            self.max_concurrent_tasks = max_concurrent_tasks
        
        # Enable performance monitoring if requested
        self.enable_performance_monitoring = enable_performance_monitoring
        if enable_performance_monitoring:
            self.resource_optimizer.start_optimization()
        
        # Task queue and results
        self.tasks = []
        self.task_queue = queue.Queue()
        self.results = {}
        
        # Processing state
        self.is_processing = False
        self.stop_requested = False
        self.progress_callback = None
        
        logger.info(f"Batch processor initialized with max {self.max_concurrent_tasks} concurrent tasks")
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if self.enable_performance_monitoring:
            self.resource_optimizer.stop_optimization()
    
    @timed("batch_processor_add_files")
    def add_files(self, file_paths: List[str], options: Dict[str, Any] = None) -> int:
        """Add files to the batch processing queue.
        
        Args:
            file_paths: List of file paths to process
            options: Processing options to apply to all files
            
        Returns:
            Number of files added
        """
        # Create tasks for each file
        for file_path in file_paths:
            task = BatchProcessingTask(file_path, options)
            self.tasks.append(task)
            self.task_queue.put(task)
        
        logger.info(f"Added {len(file_paths)} files to batch processing queue")
        return len(file_paths)
    
    @timed("batch_processor_optimize_queue")
    def optimize_queue(self):
        """Optimize the processing queue based on file sizes and system resources."""
        # Extract file information
        files = [(task.file_path, task.file_size_mb) for task in self.tasks]
        
        # Get optimized order
        optimized_files = self.batch_optimizer.optimize_batch_order(files)
        
        # Create new task queue with optimized order
        new_queue = queue.Queue()
        file_to_task = {task.file_path: task for task in self.tasks}
        
        for file_path, _ in optimized_files:
            if file_path in file_to_task:
                new_queue.put(file_to_task[file_path])
        
        # Replace the old queue
        self.task_queue = new_queue
        
        logger.info("Batch processing queue optimized")
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set a callback function to report progress.
        
        Args:
            callback: Function to call with progress updates
        """
        self.progress_callback = callback
    
    def _report_progress(self, current_task: Optional[BatchProcessingTask] = None):
        """Report progress to the callback function.
        
        Args:
            current_task: Currently processing task (if any)
        """
        if not self.progress_callback:
            return
        
        # Count tasks by status
        total = len(self.tasks)
        completed = sum(1 for task in self.tasks if task.status == "completed")
        failed = sum(1 for task in self.tasks if task.status == "failed")
        pending = sum(1 for task in self.tasks if task.status == "pending")
        processing = sum(1 for task in self.tasks if task.status == "processing")
        
        # Calculate progress percentage
        progress_percent = (completed + failed) / total * 100 if total > 0 else 0
        
        # Create progress report
        progress = {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "processing": processing,
            "progress_percent": progress_percent,
            "current_task": str(current_task) if current_task else None,
            "is_processing": self.is_processing,
            "stop_requested": self.stop_requested
        }
        
        # Add performance metrics if monitoring is enabled
        if self.enable_performance_monitoring:
            progress["performance"] = self.performance_monitor.get_metrics()
        
        # Call the callback
        try:
            self.progress_callback(progress)
        except Exception as e:
            logger.error(f"Error in progress callback: {str(e)}")
    
    @timed("batch_processor_process_task")
    def _process_task(self, task: BatchProcessingTask) -> Dict[str, Any]:
        """Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            Processing result
            
        Raises:
            ProcessingError: If processing fails
        """
        logger.info(f"Processing file: {task.file_path}")
        task.mark_started()
        self._report_progress(task)
        
        try:
            # Create a processor with optimized settings
            processor = AudioProcessor(
                enable_chunked_processing=True,
                chunk_size_seconds=self.resource_optimizer.get_optimal_chunk_size(task.file_size_mb)
            )
            
            # Load the audio file
            audio_data = self.file_handler.load_file(task.file_path)
            
            # Process the audio
            result = processor.process_audio(
                audio_path=task.file_path,
                audio_data=audio_data,
                **task.options
            )
            
            # Store results in database if available
            if self.data_manager and result:
                recording_id = self._store_results(task.file_path, audio_data, result)
                result["recording_id"] = recording_id
            
            # Mark task as completed
            task.mark_completed(result)
            logger.info(f"Completed processing file: {task.file_path}")
            
            return result
            
        except Exception as e:
            # Convert to ProcessingError if it's not already
            if not isinstance(e, VSATError):
                error = ProcessingError(
                    f"Error processing file {task.file_path}: {str(e)}",
                    context={"file_path": task.file_path, "original_error": str(e)}
                )
            else:
                error = e
            
            # Mark task as failed
            task.mark_failed(error)
            logger.error(f"Failed to process file: {task.file_path} - {str(error)}")
            
            # Re-raise the error
            raise error
    
    def _store_results(self, file_path: str, audio_data: Any, result: Dict[str, Any]) -> Optional[int]:
        """Store processing results in the database.
        
        Args:
            file_path: Path to the processed file
            audio_data: Audio data object
            result: Processing result
            
        Returns:
            Recording ID if successful, None otherwise
        """
        try:
            # Add recording to database
            recording_id = self.data_manager.add_recording(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                duration=audio_data.duration,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels
            )
            
            # Add transcript segments if available
            if "transcription" in result and "segments" in result["transcription"]:
                for segment in result["transcription"]["segments"]:
                    self.data_manager.add_transcript_segment(
                        recording_id=recording_id,
                        start_time=segment["start"],
                        end_time=segment["end"],
                        text=segment["text"],
                        speaker_id=segment.get("speaker_id", None)
                    )
            
            # Add speaker information if available
            if "diarization" in result and "speakers" in result["diarization"]:
                for speaker in result["diarization"]["speakers"]:
                    if "voice_print" in speaker:
                        self.data_manager.add_speaker(
                            name=speaker.get("name", f"Speaker {speaker['id']}"),
                            voice_print=speaker["voice_print"],
                            meta_data={"confidence": speaker.get("confidence", 0.0)}
                        )
            
            logger.info(f"Stored results for {file_path} in database (recording_id: {recording_id})")
            return recording_id
            
        except Exception as e:
            logger.error(f"Failed to store results for {file_path}: {str(e)}")
            return None
    
    @timed("batch_processor_process_all")
    def process_all(self, optimize_queue: bool = True) -> Dict[str, Any]:
        """Process all files in the batch.
        
        Args:
            optimize_queue: Whether to optimize the processing queue
            
        Returns:
            Dictionary with processing results and statistics
        """
        if self.is_processing:
            raise ProcessingError("Batch processing is already in progress")
        
        # Reset state
        self.is_processing = True
        self.stop_requested = False
        self.results = {}
        
        # Optimize queue if requested
        if optimize_queue:
            self.optimize_queue()
        
        # Reset performance metrics
        if self.enable_performance_monitoring:
            self.performance_monitor.reset_metrics()
        
        # Start time
        start_time = time.time()
        
        try:
            # Create thread pool
            with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
                # Submit initial batch of tasks
                futures = {}
                
                while not self.task_queue.empty() or futures:
                    # Submit new tasks if we have capacity and tasks available
                    while len(futures) < self.max_concurrent_tasks and not self.task_queue.empty() and not self.stop_requested:
                        task = self.task_queue.get()
                        future = executor.submit(self._process_task, task)
                        futures[future] = task
                    
                    # Check for completed tasks
                    completed_futures = []
                    for future in list(futures.keys()):
                        if future.done():
                            task = futures[future]
                            try:
                                result = future.result()
                                self.results[task.file_path] = result
                            except Exception as e:
                                # Error already handled in _process_task
                                pass
                            
                            # Report progress
                            self._report_progress()
                            
                            # Remove from active futures
                            completed_futures.append(future)
                    
                    # Remove completed futures
                    for future in completed_futures:
                        del futures[future]
                    
                    # Check if stop requested
                    if self.stop_requested:
                        logger.info("Stop requested, cancelling remaining tasks")
                        break
                    
                    # Short sleep to prevent CPU spinning
                    time.sleep(0.1)
            
            # Calculate statistics
            end_time = time.time()
            total_time = end_time - start_time
            
            completed_tasks = [task for task in self.tasks if task.status == "completed"]
            failed_tasks = [task for task in self.tasks if task.status == "failed"]
            
            # Create result summary
            summary = {
                "total_files": len(self.tasks),
                "completed_files": len(completed_tasks),
                "failed_files": len(failed_tasks),
                "total_time": total_time,
                "success_rate": len(completed_tasks) / len(self.tasks) if self.tasks else 0,
                "results": self.results
            }
            
            # Add performance metrics if monitoring is enabled
            if self.enable_performance_monitoring:
                summary["performance"] = self.performance_monitor.get_metrics()
                summary["performance_report"] = self.performance_monitor.generate_report()
            
            logger.info(f"Batch processing completed: {len(completed_tasks)}/{len(self.tasks)} files successful")
            
            return summary
            
        finally:
            self.is_processing = False
            self._report_progress()
    
    def stop(self):
        """Request to stop batch processing."""
        if self.is_processing:
            self.stop_requested = True
            logger.info("Stop requested for batch processing")
    
    def get_task_status(self) -> Dict[str, List[BatchProcessingTask]]:
        """Get the status of all tasks.
        
        Returns:
            Dictionary with tasks grouped by status
        """
        status = {
            "pending": [],
            "processing": [],
            "completed": [],
            "failed": []
        }
        
        for task in self.tasks:
            status[task.status].append(task)
        
        return status
    
    def get_results(self) -> Dict[str, Any]:
        """Get the processing results.
        
        Returns:
            Dictionary with results for each file
        """
        return self.results
    
    def clear(self):
        """Clear all tasks and results."""
        self.tasks = []
        self.task_queue = queue.Queue()
        self.results = {}
        logger.info("Batch processor cleared")
    
    def generate_report(self) -> str:
        """Generate a human-readable report of the batch processing.
        
        Returns:
            Formatted report
        """
        # Get task status
        status = self.get_task_status()
        
        # Calculate statistics
        total_files = len(self.tasks)
        completed_files = len(status["completed"])
        failed_files = len(status["failed"])
        success_rate = completed_files / total_files if total_files > 0 else 0
        
        # Calculate total processing time
        total_processing_time = sum(
            task.processing_time for task in self.tasks 
            if task.processing_time is not None
        )
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("VSAT Batch Processing Report")
        report.append("=" * 80)
        report.append("")
        
        # Summary section
        report.append("Summary:")
        report.append("-" * 40)
        report.append(f"Total files:     {total_files}")
        report.append(f"Completed files: {completed_files}")
        report.append(f"Failed files:    {failed_files}")
        report.append(f"Success rate:    {success_rate:.2%}")
        report.append(f"Total processing time: {total_processing_time:.2f} seconds")
        report.append("")
        
        # Failed files section
        if failed_files > 0:
            report.append("Failed Files:")
            report.append("-" * 40)
            for task in status["failed"]:
                error_msg = str(task.error) if task.error else "Unknown error"
                report.append(f"- {task.file_path}: {error_msg}")
            report.append("")
        
        # Performance section if monitoring is enabled
        if self.enable_performance_monitoring:
            report.append(self.performance_monitor.generate_report())
        
        return "\n".join(report) 