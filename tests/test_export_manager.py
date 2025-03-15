"""
Tests for export manager functionality.

This module contains tests to verify that the export manager works correctly.
"""

import os
import sys
import unittest
import tempfile
import json
import csv
import numpy as np
import soundfile as sf
from pathlib import Path

from src.export.export_manager import ExportManager


class TestExportManager(unittest.TestCase):
    """Test case for export manager functionality."""
    
    def setUp(self):
        """Set up the test."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test audio file
        self.audio_file = os.path.join(self.temp_dir.name, "test_audio.wav")
        self.create_test_audio_file()
        
        # Create test segments
        self.segments = self.create_test_segments()
        
        # Create export manager
        self.export_manager = ExportManager()
    
    def tearDown(self):
        """Clean up after the test."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def create_test_audio_file(self):
        """Create a test audio file with a sine wave."""
        # Generate a 440 Hz sine wave
        sample_rate = 16000
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Save as WAV file
        sf.write(self.audio_file, audio_data, sample_rate)
    
    def create_test_segments(self):
        """Create test transcript segments."""
        return [
            {
                'start': 0.0,
                'end': 1.5,
                'text': 'This is the first segment.',
                'speaker': 1,
                'speaker_name': 'Speaker 1',
                'words': [
                    {'text': 'This', 'start': 0.0, 'end': 0.3},
                    {'text': 'is', 'start': 0.3, 'end': 0.5},
                    {'text': 'the', 'start': 0.5, 'end': 0.7},
                    {'text': 'first', 'start': 0.7, 'end': 1.0},
                    {'text': 'segment.', 'start': 1.0, 'end': 1.5}
                ]
            },
            {
                'start': 1.5,
                'end': 3.0,
                'text': 'This is the second segment.',
                'speaker': 2,
                'speaker_name': 'Speaker 2',
                'words': [
                    {'text': 'This', 'start': 1.5, 'end': 1.8},
                    {'text': 'is', 'start': 1.8, 'end': 2.0},
                    {'text': 'the', 'start': 2.0, 'end': 2.2},
                    {'text': 'second', 'start': 2.2, 'end': 2.5},
                    {'text': 'segment.', 'start': 2.5, 'end': 3.0}
                ]
            },
            {
                'start': 3.0,
                'end': 4.5,
                'text': 'This is the third segment.',
                'speaker': 1,
                'speaker_name': 'Speaker 1',
                'words': [
                    {'text': 'This', 'start': 3.0, 'end': 3.3},
                    {'text': 'is', 'start': 3.3, 'end': 3.5},
                    {'text': 'the', 'start': 3.5, 'end': 3.7},
                    {'text': 'third', 'start': 3.7, 'end': 4.0},
                    {'text': 'segment.', 'start': 4.0, 'end': 4.5}
                ]
            }
        ]
    
    def test_export_transcript_txt(self):
        """Test exporting transcript to a text file."""
        # Create output file path
        output_path = os.path.join(self.temp_dir.name, "transcript.txt")
        
        # Export transcript
        result = self.export_manager.export_transcript(
            self.segments,
            output_path,
            format_type='txt',
            include_speaker=True,
            include_timestamps=True
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file contents
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check that all segments are included
            self.assertIn("Speaker 1:", content)
            self.assertIn("Speaker 2:", content)
            self.assertIn("This is the first segment.", content)
            self.assertIn("This is the second segment.", content)
            self.assertIn("This is the third segment.", content)
    
    def test_export_transcript_srt(self):
        """Test exporting transcript to an SRT file."""
        # Create output file path
        output_path = os.path.join(self.temp_dir.name, "transcript.srt")
        
        # Export transcript
        result = self.export_manager.export_transcript(
            self.segments,
            output_path,
            format_type='srt'
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file contents
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check that all segments are included
            self.assertIn("1", content)
            self.assertIn("2", content)
            self.assertIn("3", content)
            self.assertIn("-->", content)
            self.assertIn("Speaker 1: This is the first segment.", content)
            self.assertIn("Speaker 2: This is the second segment.", content)
            self.assertIn("Speaker 1: This is the third segment.", content)
    
    def test_export_transcript_json(self):
        """Test exporting transcript to a JSON file."""
        # Create output file path
        output_path = os.path.join(self.temp_dir.name, "transcript.json")
        
        # Export transcript
        result = self.export_manager.export_transcript(
            self.segments,
            output_path,
            format_type='json'
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file contents
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Check that all segments are included
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0]['speaker'], 1)
            self.assertEqual(data[1]['speaker'], 2)
            self.assertEqual(data[0]['text'], "This is the first segment.")
            self.assertEqual(data[1]['text'], "This is the second segment.")
            self.assertEqual(data[2]['text'], "This is the third segment.")
    
    def test_export_transcript_csv(self):
        """Test exporting transcript to a CSV file."""
        # Create output file path
        output_path = os.path.join(self.temp_dir.name, "transcript.csv")
        
        # Export transcript
        result = self.export_manager.export_transcript(
            self.segments,
            output_path,
            format_type='csv'
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file contents
        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Check that all segments are included
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['speaker'], '1')
            self.assertEqual(rows[1]['speaker'], '2')
            self.assertEqual(rows[0]['text'], "This is the first segment.")
            self.assertEqual(rows[1]['text'], "This is the second segment.")
            self.assertEqual(rows[2]['text'], "This is the third segment.")
    
    def test_export_audio_segment(self):
        """Test exporting an audio segment."""
        # Create output file path
        output_path = os.path.join(self.temp_dir.name, "segment.wav")
        
        # Export audio segment
        result = self.export_manager.export_audio_segment(
            self.audio_file,
            output_path,
            start=1.0,
            end=2.0,
            format_type='wav'
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file properties
        audio_data, sample_rate = sf.read(output_path)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(len(audio_data), 16000)  # 1 second at 16kHz
    
    def test_export_speaker_audio(self):
        """Test exporting speaker audio."""
        # Create output directory
        output_dir = os.path.join(self.temp_dir.name, "speaker_export")
        
        # Export speaker audio
        result = self.export_manager.export_speaker_audio(
            self.audio_file,
            self.segments,
            output_dir,
            speaker_id=1,
            format_type='wav'
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_dir))
        
        # Check that files were created
        files = os.listdir(output_dir)
        self.assertEqual(len(files), 2)  # Speaker 1 has 2 segments
        
        # Check file names
        self.assertTrue(any(f.startswith("Speaker_1_segment_001") for f in files))
        self.assertTrue(any(f.startswith("Speaker_1_segment_002") for f in files))
    
    def test_export_word_audio(self):
        """Test exporting word audio."""
        # Create output file path
        output_path = os.path.join(self.temp_dir.name, "word.wav")
        
        # Get a word from the segments
        word = self.segments[0]['words'][0]  # "This" from first segment
        
        # Export word audio
        result = self.export_manager.export_word_audio(
            self.audio_file,
            word,
            output_path,
            format_type='wav',
            padding_ms=50
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file properties
        audio_data, sample_rate = sf.read(output_path)
        self.assertEqual(sample_rate, 16000)
        
        # Expected duration: word duration (0.3s) + padding (0.05s * 2) = 0.4s
        expected_samples = int(0.4 * sample_rate)
        self.assertAlmostEqual(len(audio_data), expected_samples, delta=100)
    
    def test_export_selection(self):
        """Test exporting a selection of words."""
        # Create output file path
        output_path = os.path.join(self.temp_dir.name, "selection.wav")
        
        # Get words from the segments
        words = self.segments[0]['words'][1:4]  # "is the first" from first segment
        
        # Export selection
        result = self.export_manager.export_selection(
            self.audio_file,
            words,
            output_path,
            format_type='wav',
            include_transcript=True
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Check that transcript file was created
        transcript_path = os.path.join(self.temp_dir.name, "selection.txt")
        self.assertTrue(os.path.exists(transcript_path))
        
        # Check transcript content
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertEqual(content, "is the first")
        
        # Check audio file properties
        audio_data, sample_rate = sf.read(output_path)
        self.assertEqual(sample_rate, 16000)
        
        # Expected duration: from 0.3s to 1.0s = 0.7s
        expected_samples = int(0.7 * sample_rate)
        self.assertAlmostEqual(len(audio_data), expected_samples, delta=100)


if __name__ == "__main__":
    unittest.main() 