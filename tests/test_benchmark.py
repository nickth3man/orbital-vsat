"""
Tests for the benchmarking functionality.
"""

import os
import tempfile
import unittest
import json
import csv
from pathlib import Path
import shutil
import numpy as np

from src.evaluation.benchmark import Benchmark
from src.audio.processor import AudioProcessor
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.ml.diarization.diarizer import Diarizer

class TestBenchmark(unittest.TestCase):
    """Tests for the benchmarking functionality."""

    def setUp(self):
        """Set up test resources."""
        # Create temporary directories for test data and results
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.test_dir, "dataset")
        self.output_dir = os.path.join(self.test_dir, "results")
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test files for transcription benchmark
        self._create_transcription_test_data()
        
        # Create test files for diarization benchmark
        self._create_diarization_test_data()
        
        # Create test files for separation benchmark
        self._create_separation_test_data()
        
        # Create benchmark instance
        self.benchmark = Benchmark(output_dir=self.output_dir)

    def tearDown(self):
        """Clean up test resources."""
        shutil.rmtree(self.test_dir)

    def _create_transcription_test_data(self):
        """Create test data for transcription benchmark."""
        # Create transcription dataset directory
        trans_dir = os.path.join(self.dataset_dir, "transcription")
        os.makedirs(trans_dir, exist_ok=True)
        
        # Create a simple sine wave audio file
        duration = 3  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as WAV file
        import scipy.io.wavfile as wavfile
        wavfile.write(os.path.join(trans_dir, "test1.wav"), sample_rate, audio.astype(np.float32))
        
        # Create a transcript file
        with open(os.path.join(trans_dir, "test1.txt"), "w") as f:
            f.write("this is a test transcript")
        
        # Create metadata CSV
        with open(os.path.join(trans_dir, "metadata.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["audio_filename", "transcript"])
            writer.writerow(["test1.wav", "this is a test transcript"])

    def _create_diarization_test_data(self):
        """Create test data for diarization benchmark."""
        # Create diarization dataset directory
        diar_dir = os.path.join(self.dataset_dir, "diarization")
        os.makedirs(diar_dir, exist_ok=True)
        
        # Create a simple audio file
        duration = 5  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio[int(sample_rate * 2.5):] = 0.5 * np.sin(2 * np.pi * 880 * t[int(sample_rate * 2.5):])  # 880 Hz in second half
        
        # Save as WAV file
        import scipy.io.wavfile as wavfile
        wavfile.write(os.path.join(diar_dir, "meeting1.wav"), sample_rate, audio.astype(np.float32))
        
        # Create a RTTM file (standard diarization annotation format)
        # Format: SPEAKER meeting1 1 0.0 2.5 <NA> <NA> speaker1 <NA> <NA>
        # Format: SPEAKER meeting1 1 2.5 2.5 <NA> <NA> speaker2 <NA> <NA>
        with open(os.path.join(diar_dir, "meeting1.rttm"), "w") as f:
            f.write("SPEAKER meeting1 1 0.0 2.5 <NA> <NA> speaker1 <NA> <NA>\n")
            f.write("SPEAKER meeting1 1 2.5 2.5 <NA> <NA> speaker2 <NA> <NA>\n")

    def _create_separation_test_data(self):
        """Create test data for separation benchmark."""
        # Create separation dataset directory
        sep_dir = os.path.join(self.dataset_dir, "separation")
        os.makedirs(sep_dir, exist_ok=True)
        
        # Create two simple source audio signals
        duration = 3  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        source1 = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        source2 = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz sine wave
        
        # Create mixed audio
        mixed = source1 + source2
        
        # Normalize to prevent clipping
        mixed = mixed / max(abs(mixed))
        
        # Save audio files
        import scipy.io.wavfile as wavfile
        wavfile.write(os.path.join(sep_dir, "mixed_test1.wav"), sample_rate, mixed.astype(np.float32))
        wavfile.write(os.path.join(sep_dir, "source_test1_1.wav"), sample_rate, source1.astype(np.float32))
        wavfile.write(os.path.join(sep_dir, "source_test1_2.wav"), sample_rate, source2.astype(np.float32))
        
        # Create metadata CSV
        with open(os.path.join(sep_dir, "metadata.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mixed_filename", "source_filenames"])
            writer.writerow(["mixed_test1.wav", "source_test1_1.wav;source_test1_2.wav"])

    def test_benchmark_transcription(self):
        """Test benchmarking transcription performance."""
        # Mock the transcribe method to return a fixed transcript
        original_transcribe = WhisperTranscriber.transcribe
        
        def mock_transcribe(self, audio_file, **kwargs):
            return {
                "text": "this is a test transcript",
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": "this is a test transcript"}
                ]
            }
        
        WhisperTranscriber.transcribe = mock_transcribe
        
        try:
            # Run transcription benchmark
            results = self.benchmark.benchmark_transcription(
                dataset_path=os.path.join(self.dataset_dir, "transcription"),
                model_size="tiny",
                device="cpu"
            )
            
            # Check results
            self.assertIn("average_wer", results)
            self.assertIn("results_file", results)
            
            # Check that the results file exists
            self.assertTrue(os.path.exists(results["results_file"]))
            
            # Since we mocked the transcribe method to return the exact transcript,
            # the WER should be 0
            self.assertEqual(results["average_wer"], 0.0)
            
        finally:
            # Restore original method
            WhisperTranscriber.transcribe = original_transcribe

    def test_benchmark_diarization(self):
        """Test benchmarking diarization performance."""
        # Mock the diarize method to return fixed segments
        original_diarize = Diarizer.diarize
        
        def mock_diarize(self, audio_file, **kwargs):
            return [
                {"start": 0.0, "end": 2.5, "speaker": "speaker1"},
                {"start": 2.5, "end": 5.0, "speaker": "speaker2"}
            ]
        
        Diarizer.diarize = mock_diarize
        
        try:
            # Run diarization benchmark
            results = self.benchmark.benchmark_diarization(
                dataset_path=os.path.join(self.dataset_dir, "diarization"),
                collar=0.25,
                device="cpu"
            )
            
            # Check results
            self.assertIn("average_der", results)
            self.assertIn("results_file", results)
            
            # Check that the results file exists
            self.assertTrue(os.path.exists(results["results_file"]))
            
            # Since we mocked the diarize method to return perfect segments,
            # the DER should be 0
            self.assertEqual(results["average_der"], 0.0)
            
        finally:
            # Restore original method
            Diarizer.diarize = original_diarize

    def test_benchmark_separation(self):
        """Test benchmarking separation performance."""
        # Mock the separate method to return the source signals directly
        original_separate = AudioProcessor.separate_sources
        
        def mock_separate(self, audio_file, **kwargs):
            # Simply return the source files we know exist
            return [
                os.path.join(self.dataset_dir, "separation", "source_test1_1.wav"),
                os.path.join(self.dataset_dir, "separation", "source_test1_2.wav")
            ]
        
        AudioProcessor.separate_sources = mock_separate
        
        try:
            # Run separation benchmark
            results = self.benchmark.benchmark_separation(
                dataset_path=os.path.join(self.dataset_dir, "separation"),
                device="cpu"
            )
            
            # Check results
            self.assertIn("average_sdr", results)
            self.assertIn("results_file", results)
            
            # Check that the results file exists
            self.assertTrue(os.path.exists(results["results_file"]))
            
            # Since we're using the exact same files, SDR should be very high (nearly infinite)
            # but due to numerical precision it might not be exactly INF
            self.assertGreater(results["average_sdr"], 50.0)  # A very high SDR threshold
            
        finally:
            # Restore original method
            AudioProcessor.separate_sources = original_separate

    def test_load_datasets(self):
        """Test loading datasets for benchmarking."""
        # Test loading transcription dataset
        transcription_data = self.benchmark._load_transcription_dataset(
            os.path.join(self.dataset_dir, "transcription")
        )
        self.assertEqual(len(transcription_data), 1)
        self.assertEqual(transcription_data[0]["transcript"], "this is a test transcript")
        
        # Test loading diarization dataset
        diarization_data = self.benchmark._load_diarization_dataset(
            os.path.join(self.dataset_dir, "diarization")
        )
        self.assertEqual(len(diarization_data), 1)
        self.assertEqual(len(diarization_data[0]["reference_segments"]), 2)
        
        # Test loading separation dataset
        separation_data = self.benchmark._load_separation_dataset(
            os.path.join(self.dataset_dir, "separation")
        )
        self.assertEqual(len(separation_data), 1)
        self.assertEqual(len(separation_data[0]["source_files"]), 2)


if __name__ == "__main__":
    unittest.main() 