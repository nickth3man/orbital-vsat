"""
Runner script for VSAT benchmarks.

This script runs benchmarks for VSAT components against prepared datasets.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import Benchmark
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

def run_transcription_benchmark(args):
    """Run transcription benchmark."""
    logger.info(f"Running transcription benchmark with dataset: {args.dataset}")
    
    benchmark = Benchmark(output_dir=args.output)
    results = benchmark.benchmark_transcription(
        dataset_path=args.dataset,
        model_size=args.model_size,
        device=args.device
    )
    
    logger.info(f"Transcription benchmark results:")
    logger.info(f"Average WER: {results['average_wer']:.2f}%")
    logger.info(f"Best WER: {results['best_wer']:.2f}% ({results['best_file']})")
    logger.info(f"Worst WER: {results['worst_wer']:.2f}% ({results['worst_file']})")
    logger.info(f"Results saved to: {results['results_file']}")

def run_diarization_benchmark(args):
    """Run diarization benchmark."""
    logger.info(f"Running diarization benchmark with dataset: {args.dataset}")
    
    benchmark = Benchmark(output_dir=args.output)
    results = benchmark.benchmark_diarization(
        dataset_path=args.dataset,
        collar=args.collar,
        device=args.device
    )
    
    logger.info(f"Diarization benchmark results:")
    logger.info(f"Average DER: {results['average_der']:.2f}%")
    logger.info(f"Best DER: {results['best_der']:.2f}% ({results['best_file']})")
    logger.info(f"Worst DER: {results['worst_der']:.2f}% ({results['worst_file']})")
    logger.info(f"Results saved to: {results['results_file']}")

def run_separation_benchmark(args):
    """Run separation benchmark."""
    logger.info(f"Running separation benchmark with dataset: {args.dataset}")
    
    benchmark = Benchmark(output_dir=args.output)
    results = benchmark.benchmark_separation(
        dataset_path=args.dataset,
        device=args.device
    )
    
    logger.info(f"Separation benchmark results:")
    logger.info(f"Average SDR: {results['average_sdr']:.2f} dB")
    logger.info(f"Best SDR: {results['best_sdr']:.2f} dB ({results['best_file']})")
    logger.info(f"Worst SDR: {results['worst_sdr']:.2f} dB ({results['worst_file']})")
    logger.info(f"Results saved to: {results['results_file']}")

def main():
    """Command-line entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="VSAT Benchmark Runner")
    
    parser.add_argument('--type', type=str, required=True, 
                        choices=['transcription', 'diarization', 'separation'],
                        help="Type of benchmark to run")
    
    parser.add_argument('--dataset', type=str, required=True,
                        help="Path to prepared dataset directory")
    
    parser.add_argument('--output', type=str, default="benchmark_results",
                        help="Output directory for benchmark results")
    
    parser.add_argument('--model-size', type=str, default="base",
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help="Model size for transcription (Whisper)")
    
    parser.add_argument('--collar', type=float, default=0.25,
                        help="Collar size in seconds for diarization evaluation")
    
    parser.add_argument('--device', type=str, default="cpu",
                        choices=['cpu', 'cuda', 'mps'],
                        help="Device to run inference on")
    
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run benchmark based on type
    if args.type == 'transcription':
        run_transcription_benchmark(args)
    elif args.type == 'diarization':
        run_diarization_benchmark(args)
    elif args.type == 'separation':
        run_separation_benchmark(args)
    else:
        logger.error(f"Unknown benchmark type: {args.type}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 