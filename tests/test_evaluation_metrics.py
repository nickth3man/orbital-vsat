"""
Tests for the evaluation metrics module.

This module contains tests for the WER, DER, and SDR calculation classes.
"""

import unittest
import numpy as np

from src.evaluation.wer import WordErrorRate
from src.evaluation.der import DiarizationErrorRate
from src.evaluation.sdr import SignalDistortionRatio

class TestWordErrorRate(unittest.TestCase):
    """Test the WordErrorRate class."""
    
    def setUp(self):
        """Set up the test case."""
        self.wer_calculator = WordErrorRate()
    
    def test_perfect_match(self):
        """Test WER calculation with perfect match."""
        reference = "This is a test sentence."
        hypothesis = "This is a test sentence."
        
        result = self.wer_calculator.calculate(reference, hypothesis)
        
        self.assertEqual(result['wer'], 0.0)
        self.assertEqual(result['substitutions'], 0)
        self.assertEqual(result['deletions'], 0)
        self.assertEqual(result['insertions'], 0)
        self.assertEqual(result['total_words'], 5)
    
    def test_substitution(self):
        """Test WER calculation with word substitution."""
        reference = "This is a test sentence."
        hypothesis = "This is a simple sentence."
        
        result = self.wer_calculator.calculate(reference, hypothesis)
        
        self.assertEqual(result['substitutions'], 1)
        self.assertEqual(result['deletions'], 0)
        self.assertEqual(result['insertions'], 0)
        self.assertEqual(result['wer'], 20.0)  # 1/5 = 0.2 * 100 = 20%
    
    def test_deletion(self):
        """Test WER calculation with word deletion."""
        reference = "This is a test sentence."
        hypothesis = "This is a sentence."
        
        result = self.wer_calculator.calculate(reference, hypothesis)
        
        self.assertEqual(result['substitutions'], 0)
        self.assertEqual(result['deletions'], 1)
        self.assertEqual(result['insertions'], 0)
        self.assertEqual(result['wer'], 20.0)  # 1/5 = 0.2 * 100 = 20%
    
    def test_insertion(self):
        """Test WER calculation with word insertion."""
        reference = "This is a sentence."
        hypothesis = "This is a test sentence."
        
        result = self.wer_calculator.calculate(reference, hypothesis)
        
        self.assertEqual(result['substitutions'], 0)
        self.assertEqual(result['deletions'], 0)
        self.assertEqual(result['insertions'], 1)
        self.assertEqual(result['wer'], 25.0)  # 1/4 = 0.25 * 100 = 25%
    
    def test_case_sensitivity(self):
        """Test WER calculation with case sensitivity."""
        reference = "This is a test sentence."
        hypothesis = "this is a test sentence."
        
        # Case insensitive (default)
        result = self.wer_calculator.calculate(reference, hypothesis)
        self.assertEqual(result['wer'], 0.0)
        
        # Case sensitive
        wer_calculator = WordErrorRate(case_sensitive=True)
        result = wer_calculator.calculate(reference, hypothesis)
        self.assertEqual(result['substitutions'], 1)
        self.assertEqual(result['wer'], 20.0)
    
    def test_punctuation_handling(self):
        """Test WER calculation with punctuation handling."""
        reference = "This is a test sentence."
        hypothesis = "This is a test sentence"
        
        # Ignore punctuation (default)
        result = self.wer_calculator.calculate(reference, hypothesis)
        self.assertEqual(result['wer'], 0.0)
        
        # Consider punctuation
        wer_calculator = WordErrorRate(ignore_punctuation=False)
        result = wer_calculator.calculate(reference, hypothesis)
        self.assertEqual(result['wer'], 20.0)  # "sentence." != "sentence"
    
    def test_benchmark(self):
        """Test WER benchmark functionality."""
        references = [
            "This is the first test.",
            "This is the second test."
        ]
        
        hypotheses = [
            "This is the first test.",
            "This is a second test."
        ]
        
        result = self.wer_calculator.benchmark(references, hypotheses)
        
        self.assertEqual(result['total_words'], 9)  # 4 + 5
        self.assertEqual(result['total_substitutions'], 1)
        self.assertEqual(result['average_wer'], 100 * 1/9)

class TestDiarizationErrorRate(unittest.TestCase):
    """Test the DiarizationErrorRate class."""
    
    def setUp(self):
        """Set up the test case."""
        self.der_calculator = DiarizationErrorRate(collar=0.0, ignore_overlaps=False)
    
    def test_perfect_match(self):
        """Test DER calculation with perfect match."""
        reference = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
        ]
        
        hypothesis = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
        ]
        
        result = self.der_calculator.calculate(reference, hypothesis)
        
        self.assertAlmostEqual(result['der'], 0.0)
        self.assertAlmostEqual(result['false_alarm'], 0.0)
        self.assertAlmostEqual(result['missed_detection'], 0.0)
        self.assertAlmostEqual(result['speaker_error'], 0.0)
        self.assertAlmostEqual(result['total_time'], 2.0)
    
    def test_speaker_confusion(self):
        """Test DER calculation with speaker confusion."""
        reference = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
        ]
        
        hypothesis = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'B'},  # Wrong speaker
            {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
        ]
        
        result = self.der_calculator.calculate(reference, hypothesis)
        
        self.assertGreater(result['der'], 0.0)
        self.assertGreater(result['speaker_error'], 0.0)
    
    def test_missed_detection(self):
        """Test DER calculation with missed detection."""
        reference = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
        ]
        
        hypothesis = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'}
            # Missing second segment
        ]
        
        result = self.der_calculator.calculate(reference, hypothesis)
        
        self.assertGreater(result['der'], 0.0)
        self.assertGreater(result['missed_detection'], 0.0)
    
    def test_false_alarm(self):
        """Test DER calculation with false alarm."""
        reference = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'}
        ]
        
        hypothesis = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'B'}  # Extra segment
        ]
        
        result = self.der_calculator.calculate(reference, hypothesis)
        
        self.assertGreater(result['der'], 0.0)
        self.assertGreater(result['false_alarm'], 0.0)
    
    def test_collar(self):
        """Test DER calculation with collar."""
        reference = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
        ]
        
        hypothesis = [
            {'start': 0.1, 'end': 0.9, 'speaker': 'A'},  # Slight misalignment
            {'start': 1.1, 'end': 2.0, 'speaker': 'B'}   # Slight misalignment
        ]
        
        # Without collar
        result_no_collar = self.der_calculator.calculate(reference, hypothesis)
        self.assertGreater(result_no_collar['der'], 0.0)
        
        # With collar
        der_calculator_with_collar = DiarizationErrorRate(collar=0.2)
        result_with_collar = der_calculator_with_collar.calculate(reference, hypothesis)
        self.assertLess(result_with_collar['der'], result_no_collar['der'])
    
    def test_benchmark(self):
        """Test DER benchmark functionality."""
        references = [
            [
                {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
                {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
            ],
            [
                {'start': 0.0, 'end': 1.0, 'speaker': 'C'},
                {'start': 1.0, 'end': 2.0, 'speaker': 'D'}
            ]
        ]
        
        hypotheses = [
            [
                {'start': 0.0, 'end': 1.0, 'speaker': 'A'},
                {'start': 1.0, 'end': 2.0, 'speaker': 'B'}
            ],
            [
                {'start': 0.0, 'end': 1.0, 'speaker': 'D'},  # Wrong speaker
                {'start': 1.0, 'end': 2.0, 'speaker': 'D'}
            ]
        ]
        
        result = self.der_calculator.benchmark(references, hypotheses)
        
        self.assertGreater(result['average_der'], 0.0)
        self.assertEqual(len(result['sample_results']), 2)

class TestSignalDistortionRatio(unittest.TestCase):
    """Test the SignalDistortionRatio class."""
    
    def setUp(self):
        """Set up the test case."""
        self.sdr_calculator = SignalDistortionRatio()
    
    def test_perfect_match(self):
        """Test SDR calculation with perfect match."""
        reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        estimate = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        result = self.sdr_calculator.calculate([reference], [estimate], 16000)
        
        self.assertEqual(result['sdr'], 100.0)  # High SDR for perfect match
    
    def test_added_noise(self):
        """Test SDR calculation with added noise."""
        np.random.seed(42)  # For reproducibility
        
        reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        noise = np.random.normal(0, 0.01, 5)
        estimate = reference + noise
        
        result = self.sdr_calculator.calculate([reference], [estimate], 16000)
        
        self.assertLess(result['sdr'], 100.0)  # SDR should be lower with noise
        self.assertGreater(result['sdr'], 10.0)  # But still reasonable
    
    def test_scaling(self):
        """Test SDR calculation with scaling."""
        reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        estimate = reference * 2.0  # Scale by factor of 2
        
        result = self.sdr_calculator.calculate([reference], [estimate], 16000)
        
        self.assertGreater(result['sdr'], 10.0)  # SDR should be high for scaling
    
    def test_window_size(self):
        """Test SDR calculation with window-based calculation."""
        # Create a 1-second signal at 16kHz
        t = np.linspace(0, 1, 16000)
        reference = np.sin(2 * np.pi * 440 * t)
        estimate = reference + np.random.normal(0, 0.01, 16000)
        
        # Calculate global SDR
        result_global = self.sdr_calculator.calculate([reference], [estimate], 16000)
        
        # Calculate windowed SDR
        sdr_calculator_windowed = SignalDistortionRatio(window_size=0.1)  # 100ms windows
        result_windowed = sdr_calculator_windowed.calculate([reference], [estimate], 16000)
        
        self.assertIsNotNone(result_global['sdr'])
        self.assertIsNotNone(result_windowed['sdr'])
    
    def test_benchmark(self):
        """Test SDR benchmark functionality."""
        # Create simple test data
        references = [
            [np.array([0.1, 0.2, 0.3, 0.4, 0.5])],
            [np.array([0.5, 0.4, 0.3, 0.2, 0.1])]
        ]
        
        estimates = [
            [np.array([0.1, 0.2, 0.3, 0.4, 0.5])],  # Perfect match
            [np.array([0.5, 0.4, 0.3, 0.2, 0.1]) + 0.01]  # Slight noise
        ]
        
        sample_rates = [16000, 16000]
        
        result = self.sdr_calculator.benchmark(references, estimates, sample_rates)
        
        self.assertIsNotNone(result['average_sdr'])
        self.assertEqual(len(result['sample_results']), 2)
    
    def test_multiple_sources(self):
        """Test SDR calculation with multiple sources."""
        references = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        ]
        
        estimates = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        ]
        
        result = self.sdr_calculator.calculate(references, estimates, 16000)
        
        self.assertEqual(len(result['sdr_sources']), 2)

if __name__ == '__main__':
    unittest.main() 