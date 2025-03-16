#!/usr/bin/env python3
"""
Tests for the batch processor module.

These tests verify that the batch processor correctly handles multiple files
and optimizes resource usage during batch processing.
"""

import os
import unittest
import tempfile
import shutil
import time
from pathlib import Path
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from src.audio.batch_processor import BatchProcessor, BatchProcessingTask
from src.database.data_manager import DataManager
from src.database.db_manager import DatabaseManager
from src.utils.performance_optimizer import get_performance_monitor

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBatchProcessor(unittest.TestCase):
    """Test the batch processor functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.temp_dir, "test_batch.db")
        
        # Create test audio files
        cls.test_files = []
        for i in range(3):
            file_path = os.path.join(cls.temp_dir, f"test_audio_{i}.wav")
            cls._create_test_audio_file(file_path, duration=2.0 + i)
            cls.test_files.append(file_path)
        
        # Initialize database
        cls.db_manager = DatabaseManager(cls.test_db_path)
        cls.db_manager.initialize_database()
        
        # Initialize data manager
        cls.data_manager = DataManager(cls.db_manager)
        
        logger.info("Batch processor test setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        # Close database connection
        cls.db_manager.close()
        
        # Remove temporary directory and all files
        shutil.rmtree(cls.temp_dir)
        
        logger.info("Batch processor test cleanup complete")
    
    @staticmethod
    def _create_test_audio_file(file_path, duration=3.0, sample_rate=16000):
        """Create a test audio file with synthetic speech-like content."""
        # Generate a simple sine wave with some amplitude modulation to simulate speech
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        
        # Create two different "speakers" with different frequencies
        speaker1 = 0.5 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
        speaker2 = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t + 1.5))
        
        # Alternate between speakers
        audio = np.zeros_like(t)
        segment_length = int(sample_rate * 1.0)  # 1-second segments
        
        for i in range(0, len(t), segment_length):
            end = min(i + segment_length, len(t))
            if (i // segment_length) % 2 == 0:
                audio[i:end] = speaker1[i:end]
            else:
                audio[i:end] = speaker2[i:end]
        
        # Write to file
        sf.write(file_path, audio, sample_rate)
        
        logger.info(f"Created test audio file: {file_path}")
        return file_path
    
    def test_batch_processing_task(self):
        """Test the BatchProcessingTask class."""
        # Create a task
        file_path = self.test_files[0]
        task = BatchProcessingTask(file_path, {"option1": "value1"})
        
        # Check initial state
        self.assertEqual(task.file_path, file_path)
        self.assertEqual(task.options, {"option1": "value1"})
        self.assertEqual(task.status, "pending")
        self.assertIsNone(task.result)
        self.assertIsNone(task.error)
        self.assertIsNone(task.start_time)
        self.assertIsNone(task.end_time)
        self.assertIsNone(task.processing_time)
        
        # Test state transitions
        task.mark_started()
        self.assertEqual(task.status, "processing")
        self.assertIsNotNone(task.start_time)
        
        result = {"key": "value"}
        task.mark_completed(result)
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.result, result)
        self.assertIsNotNone(task.end_time)
        self.assertIsNotNone(task.processing_time)
        
        # Test error handling
        task = BatchProcessingTask(file_path)
        task.mark_started()
        error = Exception("Test error")
        task.mark_failed(error)
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.error, error)
        self.assertIsNotNone(task.end_time)
        self.assertIsNotNone(task.processing_time)
        
        # Test string representation
        self.assertIn(os.path.basename(file_path), str(task))
        self.assertIn("failed", str(task))
    
    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization."""
        # Test with default parameters
        processor = BatchProcessor()
        self.assertIsNone(processor.data_manager)
        self.assertTrue(processor.enable_performance_monitoring)
        self.assertGreater(processor.max_concurrent_tasks, 0)
        
        # Test with custom parameters
        processor = BatchProcessor(
            data_manager=self.data_manager,
            max_concurrent_tasks=2,
            enable_performance_monitoring=False
        )
        self.assertEqual(processor.data_manager, self.data_manager)
        self.assertEqual(processor.max_concurrent_tasks, 2)
        self.assertFalse(processor.enable_performance_monitoring)
    
    def test_add_files(self):
        """Test adding files to the batch processor."""
        processor = BatchProcessor()
        
        # Add files
        options = {"perform_diarization": True, "perform_transcription": True}
        count = processor.add_files(self.test_files, options)
        
        # Check results
        self.assertEqual(count, len(self.test_files))
        self.assertEqual(len(processor.tasks), len(self.test_files))
        self.assertEqual(processor.task_queue.qsize(), len(self.test_files))
        
        # Check task properties
        for task in processor.tasks:
            self.assertIn(task.file_path, self.test_files)
            self.assertEqual(task.options, options)
            self.assertEqual(task.status, "pending")
    
    def test_optimize_queue(self):
        """Test queue optimization."""
        processor = BatchProcessor()
        
        # Add files with different sizes
        processor.add_files(self.test_files)
        
        # Get initial queue order
        initial_queue = list(processor.task_queue.queue)
        
        # Optimize queue
        processor.optimize_queue()
        
        # Get optimized queue order
        optimized_queue = list(processor.task_queue.queue)
        
        # Check that all tasks are still in the queue
        self.assertEqual(len(optimized_queue), len(initial_queue))
        
        # Check that the queue order has been optimized
        # (This is a bit tricky to test deterministically, so we just check that
        # the order has changed or that the files are sorted by size)
        if len(self.test_files) > 1:
            is_different_order = initial_queue != optimized_queue
            is_sorted_by_size = all(
                optimized_queue[i].file_size_mb <= optimized_queue[i+1].file_size_mb
                for i in range(len(optimized_queue) - 1)
            )
            self.assertTrue(is_different_order or is_sorted_by_size)
    
    @patch('src.audio.processor.AudioProcessor.process_audio')
    def test_process_all(self, mock_process_audio):
        """Test processing all files in the batch."""
        # Mock the process_audio method to return a dummy result
        mock_result = {
            "diarization": {"speakers": []},
            "transcription": {"segments": []}
        }
        mock_process_audio.return_value = mock_result
        
        # Create batch processor
        processor = BatchProcessor(data_manager=self.data_manager)
        
        # Add files
        processor.add_files(self.test_files)
        
        # Process all files
        result = processor.process_all()
        
        # Check results
        self.assertEqual(result["total_files"], len(self.test_files))
        self.assertEqual(result["completed_files"], len(self.test_files))
        self.assertEqual(result["failed_files"], 0)
        self.assertEqual(len(result["results"]), len(self.test_files))
        
        # Check that process_audio was called for each file
        self.assertEqual(mock_process_audio.call_count, len(self.test_files))
        
        # Check task status
        status = processor.get_task_status()
        self.assertEqual(len(status["completed"]), len(self.test_files))
        self.assertEqual(len(status["failed"]), 0)
        self.assertEqual(len(status["pending"]), 0)
        self.assertEqual(len(status["processing"]), 0)
    
    @patch('src.audio.processor.AudioProcessor.process_audio')
    def test_process_with_errors(self, mock_process_audio):
        """Test processing with errors."""
        # Mock the process_audio method to raise an exception for one file
        def mock_process(audio_path, **kwargs):
            if os.path.basename(audio_path) == os.path.basename(self.test_files[1]):
                raise Exception("Test error")
            return {"diarization": {"speakers": []}, "transcription": {"segments": []}}
        
        mock_process_audio.side_effect = mock_process
        
        # Create batch processor
        processor = BatchProcessor()
        
        # Add files
        processor.add_files(self.test_files)
        
        # Process all files
        result = processor.process_all()
        
        # Check results
        self.assertEqual(result["total_files"], len(self.test_files))
        self.assertEqual(result["completed_files"], len(self.test_files) - 1)
        self.assertEqual(result["failed_files"], 1)
        self.assertEqual(len(result["results"]), len(self.test_files) - 1)
        
        # Check task status
        status = processor.get_task_status()
        self.assertEqual(len(status["completed"]), len(self.test_files) - 1)
        self.assertEqual(len(status["failed"]), 1)
        self.assertEqual(len(status["pending"]), 0)
        self.assertEqual(len(status["processing"]), 0)
        
        # Check error message
        failed_task = status["failed"][0]
        self.assertIn("Test error", str(failed_task.error))
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        # Create a mock callback
        callback = MagicMock()
        
        # Create batch processor with mock process_audio
        with patch('src.audio.processor.AudioProcessor.process_audio') as mock_process_audio:
            mock_process_audio.return_value = {
                "diarization": {"speakers": []},
                "transcription": {"segments": []}
            }
            
            # Create batch processor
            processor = BatchProcessor()
            processor.set_progress_callback(callback)
            
            # Add files
            processor.add_files(self.test_files[:1])
            
            # Process all files
            processor.process_all()
            
            # Check that callback was called
            self.assertTrue(callback.called)
            
            # Check callback arguments
            for call_args in callback.call_args_list:
                progress = call_args[0][0]
                self.assertIn("total", progress)
                self.assertIn("completed", progress)
                self.assertIn("failed", progress)
                self.assertIn("pending", progress)
                self.assertIn("processing", progress)
                self.assertIn("progress_percent", progress)
    
    def test_stop_processing(self):
        """Test stopping batch processing."""
        # Create a slow mock process_audio
        def slow_process(*args, **kwargs):
            time.sleep(0.5)
            return {"diarization": {"speakers": []}, "transcription": {"segments": []}}
        
        with patch('src.audio.processor.AudioProcessor.process_audio') as mock_process_audio:
            mock_process_audio.side_effect = slow_process
            
            # Create batch processor
            processor = BatchProcessor(max_concurrent_tasks=1)
            
            # Add files
            processor.add_files(self.test_files)
            
            # Start processing in a separate thread
            import threading
            result = [None]
            
            def process_thread():
                result[0] = processor.process_all()
            
            thread = threading.Thread(target=process_thread)
            thread.start()
            
            # Wait a bit for processing to start
            time.sleep(0.2)
            
            # Stop processing
            processor.stop()
            
            # Wait for thread to finish
            thread.join(timeout=5.0)
            
            # Check that processing was stopped
            self.assertTrue(processor.stop_requested)
            self.assertFalse(processor.is_processing)
            
            # Check that some files were processed and some were not
            status = processor.get_task_status()
            total_processed = len(status["completed"]) + len(status["failed"])
            self.assertLess(total_processed, len(self.test_files))
    
    def test_generate_report(self):
        """Test generating a batch processing report."""
        # Create batch processor with mock process_audio
        with patch('src.audio.processor.AudioProcessor.process_audio') as mock_process_audio:
            # Make one file succeed and one file fail
            def mock_process(audio_path, **kwargs):
                if os.path.basename(audio_path) == os.path.basename(self.test_files[0]):
                    return {"diarization": {"speakers": []}, "transcription": {"segments": []}}
                else:
                    raise Exception("Test error")
            
            mock_process_audio.side_effect = mock_process
            
            # Create batch processor
            processor = BatchProcessor()
            
            # Add files
            processor.add_files(self.test_files[:2])
            
            # Process all files
            processor.process_all()
            
            # Generate report
            report = processor.generate_report()
            
            # Check report content
            self.assertIn("VSAT Batch Processing Report", report)
            self.assertIn("Total files:", report)
            self.assertIn("Completed files:", report)
            self.assertIn("Failed files:", report)
            self.assertIn("Success rate:", report)
            self.assertIn("Failed Files:", report)
            self.assertIn("Test error", report)
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Create batch processor with performance monitoring
        processor = BatchProcessor(enable_performance_monitoring=True)
        
        # Get performance monitor
        monitor = get_performance_monitor()
        
        # Reset metrics
        monitor.reset_metrics()
        
        # Add files and process with mock
        with patch('src.audio.processor.AudioProcessor.process_audio') as mock_process_audio:
            mock_process_audio.return_value = {
                "diarization": {"speakers": []},
                "transcription": {"segments": []}
            }
            
            # Add files
            processor.add_files(self.test_files[:1])
            
            # Process all files
            result = processor.process_all()
            
            # Check that performance metrics were collected
            self.assertIn("performance", result)
            metrics = result["performance"]
            
            # Check that metrics include memory and CPU usage
            self.assertIn("memory", metrics)
            self.assertIn("cpu", metrics)
            
            # Check that operation timings were recorded
            self.assertIn("batch_processor_process_all", metrics)
            
            # Check that performance report was generated
            self.assertIn("performance_report", result)
            self.assertIsInstance(result["performance_report"], str)


if __name__ == "__main__":
    unittest.main() 