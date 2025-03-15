"""
Benchmarking module for VSAT.

This module provides functionality for benchmarking various components
of the VSAT system against standard datasets and known references.
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import csv

import numpy as np
import soundfile as sf
import pandas as pd

from src.evaluation.wer import WordErrorRate
from src.evaluation.der import DiarizationErrorRate
from src.evaluation.sdr import SignalDistortionRatio
from src.audio.file_handler import AudioFileHandler
from src.audio.processor import AudioProcessor
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.ml.diarization import Diarizer

logger = logging.getLogger(__name__)

class Benchmark:
    """Class for benchmarking VSAT components."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        """Initialize the benchmark.
        
        Args:
            output_dir: Directory to store benchmark results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluation metrics
        self.wer_calculator = WordErrorRate()
        self.der_calculator = DiarizationErrorRate()
        self.sdr_calculator = SignalDistortionRatio()
        
        logger.info(f"Initialized benchmark with output directory: {output_dir}")
    
    def benchmark_transcription(self, dataset_path: str, 
                              model_size: str = "medium", 
                              device: str = "cpu") -> Dict[str, Any]:
        """Benchmark transcription performance.
        
        Args:
            dataset_path: Path to dataset directory containing audio files and reference transcripts
            model_size: Whisper model size to use
            device: Device to use for inference
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        logger.info(f"Starting transcription benchmark with model_size={model_size}, device={device}")
        start_time = time.time()
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_size=model_size, device=device)
        
        # Load dataset
        dataset = self._load_transcription_dataset(dataset_path)
        
        # Run transcription benchmark
        references = []
        hypotheses = []
        file_results = []
        
        for item in dataset:
            audio_path = item['audio_path']
            reference_text = item['reference']
            
            # Load audio
            audio, sample_rate = sf.read(audio_path)
            
            # Transcribe
            transcription_start = time.time()
            result = transcriber.transcribe(audio, sample_rate)
            transcription_time = time.time() - transcription_start
            
            # Extract transcript text
            hypothesis_text = result.get('text', '')
            
            # Calculate WER
            wer_result = self.wer_calculator.calculate(reference_text, hypothesis_text)
            
            # Add to lists for aggregate calculation
            references.append(reference_text)
            hypotheses.append(hypothesis_text)
            
            # Store file result
            file_result = {
                'file': os.path.basename(audio_path),
                'wer': wer_result['wer'],
                'substitutions': wer_result['substitutions'],
                'deletions': wer_result['deletions'],
                'insertions': wer_result['insertions'],
                'total_words': wer_result['total_words'],
                'reference': reference_text,
                'hypothesis': hypothesis_text,
                'processing_time': transcription_time
            }
            file_results.append(file_result)
        
        # Calculate aggregate WER
        aggregate_result = self.wer_calculator.benchmark(references, hypotheses)
        
        # Generate benchmark summary
        benchmark_result = {
            'model_size': model_size,
            'device': device,
            'num_files': len(dataset),
            'total_words': aggregate_result['total_words'],
            'average_wer': aggregate_result['average_wer'],
            'total_substitutions': aggregate_result['total_substitutions'],
            'total_deletions': aggregate_result['total_deletions'],
            'total_insertions': aggregate_result['total_insertions'],
            'total_time': time.time() - start_time,
            'file_results': file_results
        }
        
        # Save results
        self._save_benchmark_result('transcription', benchmark_result)
        
        logger.info(f"Completed transcription benchmark. Average WER: {aggregate_result['average_wer']:.2f}%")
        return benchmark_result
    
    def benchmark_diarization(self, dataset_path: str, 
                            collar: float = 0.25,
                            device: str = "cpu") -> Dict[str, Any]:
        """Benchmark diarization performance.
        
        Args:
            dataset_path: Path to dataset directory containing audio files and reference diarization
            collar: Collar size in seconds for DER calculation
            device: Device to use for inference
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        logger.info(f"Starting diarization benchmark with collar={collar}, device={device}")
        start_time = time.time()
        
        # Initialize diarizer
        diarizer = Diarizer(device=device)
        
        # Initialize DER calculator with specified collar
        der_calculator = DiarizationErrorRate(collar=collar)
        
        # Load dataset
        dataset = self._load_diarization_dataset(dataset_path)
        
        # Run diarization benchmark
        references_list = []
        hypotheses_list = []
        file_results = []
        
        for item in dataset:
            audio_path = item['audio_path']
            reference_segments = item['reference']
            
            # Load audio
            audio, sample_rate = sf.read(audio_path)
            
            # Run diarization
            diarization_start = time.time()
            diarization_result = diarizer.diarize(audio, sample_rate)
            diarization_time = time.time() - diarization_start
            
            # Convert diarization result to segments
            hypothesis_segments = []
            for segment in diarization_result.get('segments', []):
                hypothesis_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': segment['speaker']
                })
            
            # Calculate DER
            der_result = der_calculator.calculate(reference_segments, hypothesis_segments)
            
            # Add to lists for aggregate calculation
            references_list.append(reference_segments)
            hypotheses_list.append(hypothesis_segments)
            
            # Store file result
            file_result = {
                'file': os.path.basename(audio_path),
                'der': der_result['der'],
                'false_alarm': der_result['false_alarm'],
                'missed_detection': der_result['missed_detection'],
                'speaker_error': der_result['speaker_error'],
                'total_time': der_result['total_time'],
                'num_reference_segments': len(reference_segments),
                'num_hypothesis_segments': len(hypothesis_segments),
                'processing_time': diarization_time
            }
            file_results.append(file_result)
        
        # Calculate aggregate DER
        aggregate_result = der_calculator.benchmark(references_list, hypotheses_list)
        
        # Generate benchmark summary
        benchmark_result = {
            'collar': collar,
            'device': device,
            'num_files': len(dataset),
            'average_der': aggregate_result['average_der'],
            'total_false_alarm': aggregate_result['total_false_alarm'],
            'total_missed_detection': aggregate_result['total_missed_detection'],
            'total_speaker_error': aggregate_result['total_speaker_error'],
            'total_time': time.time() - start_time,
            'file_results': file_results
        }
        
        # Save results
        self._save_benchmark_result('diarization', benchmark_result)
        
        logger.info(f"Completed diarization benchmark. Average DER: {aggregate_result['average_der']:.2f}%")
        return benchmark_result
    
    def benchmark_separation(self, dataset_path: str, 
                           device: str = "cpu") -> Dict[str, Any]:
        """Benchmark speech separation performance.
        
        Args:
            dataset_path: Path to dataset directory containing mixed audio and reference sources
            device: Device to use for inference
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        logger.info(f"Starting separation benchmark with device={device}")
        start_time = time.time()
        
        # Initialize processor
        processor = AudioProcessor(device=device)
        
        # Load dataset
        dataset = self._load_separation_dataset(dataset_path)
        
        # Run separation benchmark
        references_list = []
        estimates_list = []
        sample_rates = []
        file_results = []
        
        for item in dataset:
            mixed_path = item['mixed_path']
            source_paths = item['source_paths']
            
            # Load mixed audio
            mixed_audio, sample_rate = sf.read(mixed_path)
            
            # Load reference sources
            reference_sources = []
            for source_path in source_paths:
                source, sr = sf.read(source_path)
                assert sr == sample_rate, "Sample rates must match for all sources"
                reference_sources.append(source)
            
            # Run separation
            separation_start = time.time()
            separation_result = processor.separate_sources(mixed_audio, sample_rate)
            separation_time = time.time() - separation_start
            
            # Extract estimated sources
            estimated_sources = separation_result.get('sources', [])
            
            # Ensure we have the same number of sources
            if len(estimated_sources) != len(reference_sources):
                logger.warning(f"Number of estimated sources ({len(estimated_sources)}) "
                             f"doesn't match reference ({len(reference_sources)})")
                continue
            
            # Calculate SDR
            sdr_result = self.sdr_calculator.calculate(reference_sources, estimated_sources, sample_rate)
            
            # Add to lists for aggregate calculation
            references_list.append(reference_sources)
            estimates_list.append(estimated_sources)
            sample_rates.append(sample_rate)
            
            # Store file result
            file_result = {
                'file': os.path.basename(mixed_path),
                'sdr': sdr_result['sdr'],
                'sdr_sources': sdr_result['sdr_sources'],
                'num_sources': len(reference_sources),
                'sample_rate': sample_rate,
                'processing_time': separation_time
            }
            file_results.append(file_result)
        
        # Calculate aggregate SDR
        aggregate_result = self.sdr_calculator.benchmark(references_list, estimates_list, sample_rates)
        
        # Generate benchmark summary
        benchmark_result = {
            'device': device,
            'num_files': len(dataset),
            'average_sdr': aggregate_result['average_sdr'],
            'num_samples': aggregate_result['num_samples'],
            'num_valid_sources': aggregate_result['num_valid_sources'],
            'total_time': time.time() - start_time,
            'file_results': file_results
        }
        
        # Save results
        self._save_benchmark_result('separation', benchmark_result)
        
        logger.info(f"Completed separation benchmark. Average SDR: {aggregate_result['average_sdr']:.2f} dB")
        return benchmark_result
    
    def _load_transcription_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load transcription dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List[Dict[str, Any]]: Dataset items with audio_path and reference keys
        """
        dataset_path = Path(dataset_path)
        dataset = []
        
        # Check if dataset has a metadata file
        metadata_path = dataset_path / "metadata.csv"
        if metadata_path.exists():
            # Load metadata from CSV
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    audio_path = dataset_path / row['audio_filename']
                    if audio_path.exists():
                        dataset.append({
                            'audio_path': str(audio_path),
                            'reference': row['transcript']
                        })
        else:
            # Look for audio files with matching .txt files
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac']:
                audio_files.extend(list(dataset_path.glob(f"*{ext}")))
            
            for audio_path in audio_files:
                txt_path = audio_path.with_suffix('.txt')
                if txt_path.exists():
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        reference = f.read().strip()
                    
                    dataset.append({
                        'audio_path': str(audio_path),
                        'reference': reference
                    })
        
        logger.info(f"Loaded transcription dataset with {len(dataset)} items from {dataset_path}")
        return dataset
    
    def _load_diarization_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load diarization dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List[Dict[str, Any]]: Dataset items with audio_path and reference keys
        """
        dataset_path = Path(dataset_path)
        dataset = []
        
        # Look for audio files with matching .rttm or .json files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(list(dataset_path.glob(f"*{ext}")))
        
        for audio_path in audio_files:
            # Check for JSON reference
            json_path = audio_path.with_suffix('.json')
            rttm_path = audio_path.with_suffix('.rttm')
            
            reference_segments = []
            
            if json_path.exists():
                # Load reference segments from JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    reference_segments = json.load(f)
            elif rttm_path.exists():
                # Load reference segments from RTTM
                reference_segments = self._load_rttm(rttm_path)
            else:
                continue
            
            dataset.append({
                'audio_path': str(audio_path),
                'reference': reference_segments
            })
        
        logger.info(f"Loaded diarization dataset with {len(dataset)} items from {dataset_path}")
        return dataset
    
    def _load_rttm(self, rttm_path: str) -> List[Dict[str, Any]]:
        """Load segments from RTTM file.
        
        Args:
            rttm_path: Path to RTTM file
            
        Returns:
            List[Dict[str, Any]]: Segments with start, end, and speaker keys
        """
        segments = []
        
        with open(rttm_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    # RTTM format: Type File Channel Start Duration Speaker ...
                    segment_type = parts[0]
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    
                    if segment_type == "SPEAKER":
                        segments.append({
                            'start': start,
                            'end': start + duration,
                            'speaker': speaker
                        })
        
        return segments
    
    def _load_separation_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load separation dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List[Dict[str, Any]]: Dataset items with mixed_path and source_paths keys
        """
        dataset_path = Path(dataset_path)
        dataset = []
        
        # Check if dataset has a metadata file
        metadata_path = dataset_path / "metadata.csv"
        if metadata_path.exists():
            # Load metadata from CSV
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mixed_path = dataset_path / row['mixed_filename']
                    source_paths = [dataset_path / path for path in row['source_filenames'].split(';')]
                    
                    if mixed_path.exists() and all(path.exists() for path in source_paths):
                        dataset.append({
                            'mixed_path': str(mixed_path),
                            'source_paths': [str(path) for path in source_paths]
                        })
        else:
            # Look for mixed_*.wav files with matching source_*_*.wav files
            mixed_files = list(dataset_path.glob("mixed_*.wav"))
            
            for mixed_path in mixed_files:
                # Extract ID from filename (e.g., mixed_001.wav -> 001)
                file_id = mixed_path.stem.split('_')[1]
                
                # Look for source files with matching ID
                source_paths = list(dataset_path.glob(f"source_{file_id}_*.wav"))
                
                if source_paths:
                    dataset.append({
                        'mixed_path': str(mixed_path),
                        'source_paths': [str(path) for path in source_paths]
                    })
        
        logger.info(f"Loaded separation dataset with {len(dataset)} items from {dataset_path}")
        return dataset
    
    def _save_benchmark_result(self, benchmark_type: str, result: Dict[str, Any]) -> None:
        """Save benchmark result to file.
        
        Args:
            benchmark_type: Type of benchmark (transcription, diarization, separation)
            result: Benchmark result dictionary
        """
        # Create timestamp for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        benchmark_dir = os.path.join(self.output_dir, benchmark_type)
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Save result as JSON
        json_path = os.path.join(benchmark_dir, f"{benchmark_type}_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        # Save summary as CSV
        csv_path = os.path.join(benchmark_dir, f"{benchmark_type}_{timestamp}.csv")
        
        if benchmark_type == 'transcription':
            # Create DataFrame from file results
            df = pd.DataFrame(result['file_results'])
            df.to_csv(csv_path, index=False)
            
            # Create summary file
            summary_path = os.path.join(benchmark_dir, f"{benchmark_type}_{timestamp}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcription Benchmark Summary\n")
                f.write(f"------------------------------\n")
                f.write(f"Model Size: {result['model_size']}\n")
                f.write(f"Device: {result['device']}\n")
                f.write(f"Number of Files: {result['num_files']}\n")
                f.write(f"Total Words: {result['total_words']}\n")
                f.write(f"Average WER: {result['average_wer']:.2f}%\n")
                f.write(f"Total Substitutions: {result['total_substitutions']}\n")
                f.write(f"Total Deletions: {result['total_deletions']}\n")
                f.write(f"Total Insertions: {result['total_insertions']}\n")
                f.write(f"Total Processing Time: {result['total_time']:.2f} seconds\n")
        
        elif benchmark_type == 'diarization':
            # Create DataFrame from file results
            df = pd.DataFrame(result['file_results'])
            df.to_csv(csv_path, index=False)
            
            # Create summary file
            summary_path = os.path.join(benchmark_dir, f"{benchmark_type}_{timestamp}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Diarization Benchmark Summary\n")
                f.write(f"-----------------------------\n")
                f.write(f"Collar: {result['collar']} seconds\n")
                f.write(f"Device: {result['device']}\n")
                f.write(f"Number of Files: {result['num_files']}\n")
                f.write(f"Average DER: {result['average_der']:.2f}%\n")
                f.write(f"Total False Alarm: {result['total_false_alarm']:.2f} seconds\n")
                f.write(f"Total Missed Detection: {result['total_missed_detection']:.2f} seconds\n")
                f.write(f"Total Speaker Error: {result['total_speaker_error']:.2f} seconds\n")
                f.write(f"Total Processing Time: {result['total_time']:.2f} seconds\n")
        
        elif benchmark_type == 'separation':
            # Create DataFrame from file results
            file_results = []
            for i, fr in enumerate(result['file_results']):
                file_result = {
                    'file': fr['file'],
                    'sdr': fr['sdr'],
                    'num_sources': fr['num_sources'],
                    'sample_rate': fr['sample_rate'],
                    'processing_time': fr['processing_time']
                }
                
                # Add SDR for each source
                for j, sdr in enumerate(fr['sdr_sources']):
                    file_result[f'sdr_source_{j+1}'] = sdr
                
                file_results.append(file_result)
            
            df = pd.DataFrame(file_results)
            df.to_csv(csv_path, index=False)
            
            # Create summary file
            summary_path = os.path.join(benchmark_dir, f"{benchmark_type}_{timestamp}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Separation Benchmark Summary\n")
                f.write(f"----------------------------\n")
                f.write(f"Device: {result['device']}\n")
                f.write(f"Number of Files: {result['num_files']}\n")
                f.write(f"Average SDR: {result['average_sdr']:.2f} dB\n")
                f.write(f"Number of Valid Sources: {result['num_valid_sources']}\n")
                f.write(f"Total Processing Time: {result['total_time']:.2f} seconds\n")
        
        logger.info(f"Saved benchmark results to {json_path} and {csv_path}")


def main():
    """Command-line entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="VSAT Benchmarking Tool")
    
    parser.add_argument('--type', type=str, required=True,
                       choices=['transcription', 'diarization', 'separation'],
                       help="Type of benchmark to run")
    
    parser.add_argument('--dataset', type=str, required=True,
                       help="Path to dataset directory")
    
    parser.add_argument('--output', type=str, default="benchmarks",
                       help="Output directory for benchmark results")
    
    parser.add_argument('--model-size', type=str, default="medium",
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help="Model size for transcription benchmark")
    
    parser.add_argument('--collar', type=float, default=0.25,
                       help="Collar size in seconds for diarization benchmark")
    
    parser.add_argument('--device', type=str, default="cpu",
                       choices=['cpu', 'cuda'],
                       help="Device to use for inference")
    
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                       format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # Initialize benchmark
    benchmark = Benchmark(output_dir=args.output)
    
    # Run benchmark
    if args.type == 'transcription':
        benchmark.benchmark_transcription(args.dataset, args.model_size, args.device)
    elif args.type == 'diarization':
        benchmark.benchmark_diarization(args.dataset, args.collar, args.device)
    elif args.type == 'separation':
        benchmark.benchmark_separation(args.dataset, args.device)


if __name__ == "__main__":
    main() 