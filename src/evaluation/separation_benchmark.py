"""
Benchmark script for speaker separation quality metrics.

This module provides functionality to evaluate speaker separation
against the target quality metrics:
1. >10dB SDR for separated speakers
2. Minimal bleed-through between separated audio streams
3. Preservation of voice characteristics during separation
"""

import os
import sys
import time
import logging
import numpy as np
import soundfile as sf
import argparse
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from src.audio.processor import AudioProcessor
from src.evaluation.sdr import SignalDistortionRatio
from src.evaluation.visualize_results import plot_sdr_results

logger = logging.getLogger(__name__)

def run_separation_benchmark(dataset_path: str, 
                           output_dir: str, 
                           model_types: List[str] = ["conv_tasnet", "sudo_rm_rf", "kaituoxu"],
                           post_process: bool = True) -> Dict[str, Any]:
    """Run a benchmark of speaker separation quality metrics.
    
    Args:
        dataset_path: Path to dataset with mixed audio and reference sources
        output_dir: Path to output directory for results
        model_types: List of separation model types to evaluate
        post_process: Whether to apply post-processing to separated sources
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results structure
    results = {
        "overall": {},
        "models": {},
        "sample_results": []
    }
    
    # Initialize SDR calculator
    sdr_calculator = SignalDistortionRatio(use_db=True)
    
    # Find dataset files
    dataset_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("_mixed.wav"):
                # Expect format: sample_id_mixed.wav with corresponding sample_id_source1.wav, sample_id_source2.wav, etc.
                dataset_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(dataset_files)} mixed audio files in dataset")
    
    # Benchmark each model type
    for model_type in model_types:
        logger.info(f"Benchmarking model type: {model_type}")
        model_results = {
            "sdr_values": [],
            "sample_results": [],
            "processing_times": []
        }
        
        # Initialize processor
        processor = AudioProcessor()
        
        # Process each sample
        for mixed_file in dataset_files:
            # Get sample base name
            base_name = os.path.basename(mixed_file).replace("_mixed.wav", "")
            base_dir = os.path.dirname(mixed_file)
            
            # Find reference source files
            reference_files = []
            i = 1
            while True:
                source_file = os.path.join(base_dir, f"{base_name}_source{i}.wav")
                if os.path.exists(source_file):
                    reference_files.append(source_file)
                    i += 1
                else:
                    break
            
            if not reference_files:
                logger.warning(f"No reference sources found for {mixed_file}")
                continue
            
            logger.info(f"Processing {mixed_file} with {len(reference_files)} reference sources")
            
            # Load reference sources
            reference_sources = []
            reference_sample_rate = None
            for ref_file in reference_files:
                ref_audio, sr = sf.read(ref_file)
                reference_sources.append(ref_audio)
                reference_sample_rate = sr
            
            # Create output directory for separated sources
            sample_output_dir = os.path.join(output_dir, model_type, base_name)
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # Separate sources
            start_time = time.time()
            try:
                separated_files = processor.separate_sources(
                    mixed_file,
                    model_type=model_type,
                    output_dir=sample_output_dir,
                    post_process=post_process
                )
                
                # Load separated sources
                separated_sources = []
                for sep_file in separated_files:
                    sep_audio, _ = sf.read(sep_file)
                    separated_sources.append(sep_audio)
                
                # Match the number of sources if different
                while len(separated_sources) < len(reference_sources):
                    separated_sources.append(np.zeros_like(reference_sources[0]))
                
                while len(reference_sources) < len(separated_sources):
                    reference_sources.append(np.zeros_like(separated_sources[0]))
                
                # Calculate processing time
                processing_time = time.time() - start_time
                model_results["processing_times"].append(processing_time)
                
                # Calculate SDR metrics
                sdr_result = sdr_calculator.calculate(reference_sources, separated_sources, reference_sample_rate)
                model_results["sdr_values"].extend([sdr for sdr in sdr_result["sdr_sources"] if sdr != float('-inf')])
                
                # Add sample result
                sample_result = {
                    "file": base_name,
                    "model": model_type,
                    "sdr": sdr_result["sdr"],
                    "sdr_sources": sdr_result["sdr_sources"],
                    "num_sources": len(reference_sources),
                    "processing_time": processing_time,
                    "sample_rate": reference_sample_rate
                }
                model_results["sample_results"].append(sample_result)
                results["sample_results"].append(sample_result)
                
                logger.info(f"Sample {base_name} SDR: {sdr_result['sdr']:.2f} dB")
                
            except Exception as e:
                logger.error(f"Error processing {mixed_file}: {e}")
        
        # Calculate model statistics
        if model_results["sdr_values"]:
            avg_sdr = np.mean(model_results["sdr_values"])
            median_sdr = np.median(model_results["sdr_values"])
            min_sdr = np.min(model_results["sdr_values"])
            max_sdr = np.max(model_results["sdr_values"])
            std_sdr = np.std(model_results["sdr_values"])
            
            # Calculate percentage of samples meeting the >10dB SDR target
            target_met = sum(1 for sdr in model_results["sdr_values"] if sdr > 10.0)
            percent_target_met = 100 * target_met / len(model_results["sdr_values"])
            
            # Average processing time
            avg_processing_time = np.mean(model_results["processing_times"]) if model_results["processing_times"] else 0
            
            # Store statistics
            model_stats = {
                "average_sdr": avg_sdr,
                "median_sdr": median_sdr,
                "min_sdr": min_sdr,
                "max_sdr": max_sdr,
                "std_sdr": std_sdr,
                "num_samples": len(model_results["sample_results"]),
                "num_sources": len(model_results["sdr_values"]),
                "target_met_count": target_met,
                "target_met_percent": percent_target_met,
                "average_processing_time": avg_processing_time,
                "post_processing_applied": post_process
            }
            
            model_results["statistics"] = model_stats
            results["models"][model_type] = model_stats
            
            logger.info(f"Model {model_type} - Avg SDR: {avg_sdr:.2f} dB, "
                      f"Target met: {percent_target_met:.1f}%, "
                      f"Avg processing time: {avg_processing_time:.2f}s")
        else:
            logger.warning(f"No valid results for model {model_type}")
    
    # Calculate overall statistics
    all_sdrs = []
    for model_type, model_data in results["models"].items():
        all_sdrs.extend([sdr for sdr in model_data.get("sdr_values", [])])
    
    if all_sdrs:
        results["overall"] = {
            "average_sdr": np.mean(all_sdrs),
            "median_sdr": np.median(all_sdrs),
            "min_sdr": np.min(all_sdrs),
            "max_sdr": np.max(all_sdrs),
            "std_sdr": np.std(all_sdrs),
            "num_models": len(model_types),
            "num_samples": len(results["sample_results"]),
            "post_processing_applied": post_process
        }
    
    # Generate visualizations
    for model_type in model_types:
        if model_type in results["models"]:
            model_viz_dir = os.path.join(output_dir, f"{model_type}_viz")
            os.makedirs(model_viz_dir, exist_ok=True)
            
            # Extract model sample results
            model_samples = [s for s in results["sample_results"] if s["model"] == model_type]
            if model_samples:
                plot_sdr_results(model_samples, model_viz_dir)
    
    # Save results to file
    import json
    with open(os.path.join(output_dir, "separation_benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark speaker separation quality")
    parser.add_argument("--dataset", "-d", required=True, help="Path to dataset directory")
    parser.add_argument("--output", "-o", required=True, help="Path to output directory")
    parser.add_argument("--models", "-m", default="all", help="Comma-separated list of models to benchmark")
    parser.add_argument("--no-post-process", action="store_true", help="Disable post-processing")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output, "separation_benchmark.log"))
        ]
    )
    
    # Parse models
    if args.models.lower() == "all":
        models = ["conv_tasnet", "sudo_rm_rf", "kaituoxu"]
    else:
        models = args.models.split(",")
    
    # Run benchmark
    results = run_separation_benchmark(
        args.dataset,
        args.output,
        model_types=models,
        post_process=not args.no_post_process
    )
    
    # Print summary
    print("\n=== Speaker Separation Benchmark Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Models evaluated: {', '.join(models)}")
    print(f"Post-processing: {'Enabled' if not args.no_post_process else 'Disabled'}")
    print("\nModel performance (SDR in dB):")
    print("-" * 60)
    print(f"{'Model':<12} {'Avg SDR':<10} {'Target Met %':<12} {'Min/Max':<15} {'Proc. Time':<10}")
    print("-" * 60)
    
    for model, stats in results["models"].items():
        print(f"{model:<12} {stats['average_sdr']:<10.2f} {stats['target_met_percent']:<12.1f} "
              f"{stats['min_sdr']:<6.2f}/{stats['max_sdr']:<6.2f} {stats['average_processing_time']:<10.2f}")
    
    print("-" * 60)
    print(f"Overall average SDR: {results['overall']['average_sdr']:.2f} dB")
    print(f"Results saved to: {os.path.join(args.output, 'separation_benchmark_results.json')}")

if __name__ == "__main__":
    main() 