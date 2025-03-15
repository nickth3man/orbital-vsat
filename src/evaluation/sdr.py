"""
Signal-to-Distortion Ratio (SDR) calculation module.

This module provides functionality for calculating Signal-to-Distortion Ratio,
a metric for evaluating the quality of audio source separation.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class SignalDistortionRatio:
    """Class for calculating Signal-to-Distortion Ratio (SDR)."""
    
    def __init__(self, window_size: Optional[float] = None, use_db: bool = True):
        """Initialize the SDR calculator.
        
        Args:
            window_size: Window size in seconds for frame-wise SDR calculation or None for global
            use_db: Whether to return results in decibels (dB) or as a ratio
        """
        self.window_size = window_size
        self.use_db = use_db
        
        logger.debug(f"Initialized SDR calculator (window_size={window_size}s, "
                    f"use_db={use_db})")
    
    def calculate(self, references: List[np.ndarray], estimates: List[np.ndarray], 
                sample_rate: int) -> Dict[str, Any]:
        """Calculate SDR between reference and estimated sources.
        
        Args:
            references: List of reference source signals as numpy arrays
            estimates: List of estimated source signals as numpy arrays
            sample_rate: Sample rate of the audio signals
            
        Returns:
            Dict[str, Any]: Dictionary with SDR metrics including:
                - sdr: Overall SDR (average across sources)
                - sdr_sources: SDR for each source
                - sample_rate: Sample rate of the audio
        """
        # Validate inputs
        if len(references) != len(estimates):
            raise ValueError("Number of reference and estimated sources must match.")
        
        if any(ref.shape != est.shape for ref, est in zip(references, estimates)):
            raise ValueError("Shapes of reference and estimated sources must match.")
        
        # Calculate SDR for each source
        sdr_values = []
        
        for i, (reference, estimate) in enumerate(zip(references, estimates)):
            # Handle silent reference or estimate
            if np.max(np.abs(reference)) < 1e-10 or np.max(np.abs(estimate)) < 1e-10:
                logger.warning(f"Source {i} contains silent signals, setting SDR to -inf")
                sdr_values.append(float('-inf'))
                continue
            
            if self.window_size is None:
                # Global SDR
                sdr = self._calculate_sdr(reference, estimate)
                sdr_values.append(sdr)
            else:
                # Frame-wise SDR
                frame_sdrs = self._calculate_frame_wise_sdr(reference, estimate, sample_rate)
                # Average over frames
                sdr = np.mean(frame_sdrs) if frame_sdrs.size > 0 else float('-inf')
                sdr_values.append(sdr)
        
        # Calculate overall SDR (average across sources)
        valid_sdrs = [sdr for sdr in sdr_values if sdr != float('-inf')]
        overall_sdr = np.mean(valid_sdrs) if valid_sdrs else float('-inf')
        
        results = {
            'sdr': overall_sdr,
            'sdr_sources': sdr_values,
            'sample_rate': sample_rate
        }
        
        logger.debug(f"SDR calculation: {results}")
        return results
    
    def _calculate_sdr(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        """Calculate SDR between reference and estimate signals.
        
        Args:
            reference: Reference source signal
            estimate: Estimated source signal
            
        Returns:
            float: SDR value
        """
        # Calculate SDR based on BSS Eval methodology
        # SDR = 10*log10(||s_target||² / ||e_total||²)
        # Where s_target is the target signal and e_total is the total error
        
        # Ensure signals are in the right shape
        reference = reference.flatten()
        estimate = estimate.flatten()
        
        # Calculate the projection coefficient
        ref_power = np.sum(reference ** 2)
        ref_proj = np.sum(reference * estimate) / ref_power
        
        # Calculate the target signal
        s_target = ref_proj * reference
        
        # Calculate the total error
        e_total = estimate - s_target
        
        # Calculate the SDR
        target_power = np.sum(s_target ** 2)
        error_power = np.sum(e_total ** 2)
        
        if error_power < 1e-10:
            # If error is very small, return a high SDR value
            sdr = 100.0
        else:
            sdr = target_power / error_power
            
            if self.use_db:
                sdr = 10 * np.log10(sdr)
        
        return float(sdr)
    
    def _calculate_frame_wise_sdr(self, reference: np.ndarray, estimate: np.ndarray, 
                                sample_rate: int) -> np.ndarray:
        """Calculate frame-wise SDR between reference and estimate signals.
        
        Args:
            reference: Reference source signal
            estimate: Estimated source signal
            sample_rate: Sample rate of the audio signals
            
        Returns:
            np.ndarray: Array of SDR values for each frame
        """
        # Calculate window length in samples
        window_length = int(self.window_size * sample_rate)
        hop_length = window_length // 2  # 50% overlap
        
        # Ensure signals are in the right shape
        reference = reference.flatten()
        estimate = estimate.flatten()
        
        # Calculate number of frames
        num_frames = 1 + int((len(reference) - window_length) / hop_length)
        
        # Calculate SDR for each frame
        frame_sdrs = []
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + window_length
            
            ref_frame = reference[start:end]
            est_frame = estimate[start:end]
            
            # Skip frames with low energy
            if np.max(np.abs(ref_frame)) < 1e-10 or np.max(np.abs(est_frame)) < 1e-10:
                continue
            
            sdr = self._calculate_sdr(ref_frame, est_frame)
            frame_sdrs.append(sdr)
        
        return np.array(frame_sdrs)
    
    def benchmark(self, references_list: List[List[np.ndarray]], 
                estimates_list: List[List[np.ndarray]], 
                sample_rates: List[int]) -> Dict[str, Any]:
        """Run a benchmark on multiple audio source separation results.
        
        Args:
            references_list: List of lists of reference source signals
            estimates_list: List of lists of estimated source signals
            sample_rates: List of sample rates for each audio
            
        Returns:
            Dict[str, Any]: Benchmark results including average SDR and per-sample metrics
        """
        if not (len(references_list) == len(estimates_list) == len(sample_rates)):
            raise ValueError("Number of reference lists, estimate lists, and sample rates must match.")
        
        results = []
        all_sdrs = []
        
        for i, (references, estimates, sample_rate) in enumerate(
                zip(references_list, estimates_list, sample_rates)):
            result = self.calculate(references, estimates, sample_rate)
            results.append(result)
            
            # Collect valid SDRs
            valid_sdrs = [sdr for sdr in result['sdr_sources'] if sdr != float('-inf')]
            all_sdrs.extend(valid_sdrs)
        
        # Calculate overall SDR
        avg_sdr = np.mean(all_sdrs) if all_sdrs else float('-inf')
        
        benchmark_results = {
            'average_sdr': avg_sdr,
            'sample_results': results,
            'num_samples': len(results),
            'num_valid_sources': len(all_sdrs)
        }
        
        logger.info(f"SDR benchmark completed with {len(results)} samples and {len(all_sdrs)} "
                  f"valid sources. Average SDR: {avg_sdr:.2f}" + 
                  (" dB" if self.use_db else ""))
        return benchmark_results
    
    def evaluate_against_datasets(self, 
                                dataset_refs: Dict[str, List[List[np.ndarray]]],
                                dataset_ests: Dict[str, List[List[np.ndarray]]],
                                dataset_rates: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
        """Evaluate against multiple datasets.
        
        Args:
            dataset_refs: Dictionary mapping dataset names to lists of reference sources
            dataset_ests: Dictionary mapping dataset names to lists of estimated sources
            dataset_rates: Dictionary mapping dataset names to lists of sample rates
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping dataset names to benchmark results
        """
        results = {}
        
        for dataset_name in dataset_refs.keys():
            if dataset_name not in dataset_ests or dataset_name not in dataset_rates:
                logger.warning(f"Dataset '{dataset_name}' missing from estimates or sample rates")
                continue
            
            references = dataset_refs[dataset_name]
            estimates = dataset_ests[dataset_name]
            sample_rates = dataset_rates[dataset_name]
            
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            benchmark_result = self.benchmark(references, estimates, sample_rates)
            results[dataset_name] = benchmark_result
        
        return results 