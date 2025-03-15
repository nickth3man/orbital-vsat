"""
Diarization Error Rate (DER) calculation module.

This module provides functionality for calculating Diarization Error Rate,
a metric for evaluating the quality of speaker diarization systems.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class DiarizationErrorRate:
    """Class for calculating Diarization Error Rate (DER)."""
    
    def __init__(self, collar: float = 0.25, ignore_overlaps: bool = False):
        """Initialize the DER calculator.
        
        Args:
            collar: Collar size in seconds to ignore around speaker boundaries
            ignore_overlaps: Whether to ignore overlapping speech in the reference
        """
        self.collar = collar
        self.ignore_overlaps = ignore_overlaps
        
        logger.debug(f"Initialized DER calculator (collar={collar}s, "
                    f"ignore_overlaps={ignore_overlaps})")
    
    def calculate(self, reference: List[Dict[str, Any]], 
                 hypothesis: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate Diarization Error Rate between reference and hypothesis segments.
        
        Args:
            reference: List of reference segments, each with 'start', 'end', and 'speaker' keys
            hypothesis: List of hypothesis segments, each with 'start', 'end', and 'speaker' keys
            
        Returns:
            Dict[str, float]: Dictionary with DER metrics including:
                - der: Diarization Error Rate as a percentage
                - false_alarm: False alarm time in seconds
                - missed_detection: Missed detection time in seconds
                - speaker_error: Speaker error time in seconds
                - total_time: Total reference speech time in seconds
        """
        # Calculate total reference speech time
        total_time = sum(segment['end'] - segment['start'] for segment in reference)
        
        if total_time == 0:
            logger.warning("Reference contains no speech time")
            return {
                'der': 100.0,
                'false_alarm': 0.0,
                'missed_detection': 0.0,
                'speaker_error': 0.0,
                'total_time': 0.0
            }
        
        # Calculate error components
        false_alarm, missed_detection, speaker_error = self._calculate_error_components(
            reference, hypothesis
        )
        
        # Calculate DER
        der = 100.0 * (false_alarm + missed_detection + speaker_error) / total_time
        
        results = {
            'der': der,
            'false_alarm': false_alarm,
            'missed_detection': missed_detection,
            'speaker_error': speaker_error,
            'total_time': total_time
        }
        
        logger.debug(f"DER calculation: {results}")
        return results
    
    def _calculate_error_components(self, reference: List[Dict[str, Any]], 
                                  hypothesis: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """Calculate the error components for DER.
        
        Args:
            reference: List of reference segments
            hypothesis: List of hypothesis segments
            
        Returns:
            Tuple[float, float, float]: False alarm, missed detection, and speaker error time
        """
        # Convert reference and hypothesis to frame-based representation
        frame_duration = 0.01  # 10ms frames
        
        # Find latest end time for framing
        latest_end = max(
            max(segment['end'] for segment in reference) if reference else 0,
            max(segment['end'] for segment in hypothesis) if hypothesis else 0
        )
        
        n_frames = int(latest_end / frame_duration) + 1
        
        # Create frame labels
        ref_frames = self._segments_to_frames(reference, n_frames, frame_duration)
        hyp_frames = self._segments_to_frames(hypothesis, n_frames, frame_duration)
        
        # Apply collar around speaker boundaries in reference
        if self.collar > 0:
            ref_frames = self._apply_collar(reference, ref_frames, n_frames, frame_duration)
        
        # Ignore overlapping speech if requested
        if self.ignore_overlaps:
            overlap_mask = np.sum(ref_frames, axis=1) > 1
            ref_frames[:, overlap_mask] = 0
        
        # Calculate error components
        false_alarm_frames = 0
        missed_detection_frames = 0
        speaker_error_frames = 0
        
        for i in range(n_frames):
            ref_speaking = np.sum(ref_frames[:, i]) > 0
            hyp_speaking = np.sum(hyp_frames[:, i]) > 0
            
            if not ref_speaking and hyp_speaking:
                # False alarm: hypothesis has speech, but reference doesn't
                false_alarm_frames += 1
            elif ref_speaking and not hyp_speaking:
                # Missed detection: reference has speech, but hypothesis doesn't
                missed_detection_frames += 1
            elif ref_speaking and hyp_speaking:
                # Both have speech, check if speakers match
                ref_speakers = set(np.where(ref_frames[:, i] > 0)[0])
                hyp_speakers = set(np.where(hyp_frames[:, i] > 0)[0])
                
                # If speakers don't match, count as speaker error
                if ref_speakers != hyp_speakers:
                    speaker_error_frames += 1
        
        # Convert frames to seconds
        false_alarm = false_alarm_frames * frame_duration
        missed_detection = missed_detection_frames * frame_duration
        speaker_error = speaker_error_frames * frame_duration
        
        return false_alarm, missed_detection, speaker_error
    
    def _segments_to_frames(self, segments: List[Dict[str, Any]], 
                          n_frames: int, 
                          frame_duration: float) -> np.ndarray:
        """Convert segment-based representation to frame-based representation.
        
        Args:
            segments: List of segments
            n_frames: Number of frames
            frame_duration: Duration of each frame in seconds
            
        Returns:
            np.ndarray: Binary matrix where rows correspond to speakers and columns to frames
        """
        # Collect unique speakers
        speakers = list(set(segment['speaker'] for segment in segments))
        n_speakers = len(speakers)
        
        # Create speaker to index mapping
        speaker_to_idx = {speaker: i for i, speaker in enumerate(speakers)}
        
        # Create frame labels
        frames = np.zeros((n_speakers, n_frames), dtype=np.int8)
        
        for segment in segments:
            speaker_idx = speaker_to_idx[segment['speaker']]
            start_frame = max(0, int(segment['start'] / frame_duration))
            end_frame = min(n_frames, int(segment['end'] / frame_duration))
            
            frames[speaker_idx, start_frame:end_frame] = 1
        
        return frames
    
    def _apply_collar(self, segments: List[Dict[str, Any]], 
                    frames: np.ndarray, 
                    n_frames: int, 
                    frame_duration: float) -> np.ndarray:
        """Apply collar around speaker boundaries in frame-based representation.
        
        Args:
            segments: List of segments
            frames: Frame-based representation
            n_frames: Number of frames
            frame_duration: Duration of each frame in seconds
            
        Returns:
            np.ndarray: Frame-based representation with collar applied
        """
        collar_frames = int(self.collar / frame_duration)
        
        # Create a copy of frames to avoid modifying the original
        frames_with_collar = frames.copy()
        
        for segment in segments:
            start_frame = max(0, int(segment['start'] / frame_duration))
            end_frame = min(n_frames, int(segment['end'] / frame_duration))
            
            # Apply collar at the beginning of the segment
            collar_start = max(0, start_frame - collar_frames)
            frames_with_collar[:, collar_start:start_frame] = 0
            
            # Apply collar at the end of the segment
            collar_end = min(n_frames, end_frame + collar_frames)
            frames_with_collar[:, end_frame:collar_end] = 0
        
        return frames_with_collar
    
    def benchmark(self, reference_list: List[List[Dict[str, Any]]], 
                hypothesis_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Run a benchmark on multiple diarization results.
        
        Args:
            reference_list: List of reference segment lists
            hypothesis_list: List of hypothesis segment lists
            
        Returns:
            Dict[str, Any]: Benchmark results including average DER and per-sample metrics
        """
        if len(reference_list) != len(hypothesis_list):
            raise ValueError("Number of reference and hypothesis diarizations must match.")
        
        results = []
        total_time = 0.0
        total_false_alarm = 0.0
        total_missed_detection = 0.0
        total_speaker_error = 0.0
        
        for i, (reference, hypothesis) in enumerate(zip(reference_list, hypothesis_list)):
            result = self.calculate(reference, hypothesis)
            results.append(result)
            
            total_time += result['total_time']
            total_false_alarm += result['false_alarm']
            total_missed_detection += result['missed_detection']
            total_speaker_error += result['speaker_error']
        
        # Calculate aggregate DER
        if total_time == 0:
            avg_der = 100.0
        else:
            avg_der = 100.0 * (total_false_alarm + total_missed_detection + total_speaker_error) / total_time
        
        benchmark_results = {
            'average_der': avg_der,
            'total_false_alarm': total_false_alarm,
            'total_missed_detection': total_missed_detection,
            'total_speaker_error': total_speaker_error,
            'total_time': total_time,
            'sample_results': results
        }
        
        logger.info(f"DER benchmark completed with {len(results)} samples. Average DER: {avg_der:.2f}%")
        return benchmark_results 