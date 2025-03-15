"""
Transcript export functionality for VSAT.

This module provides functionality for exporting transcripts in various formats.
"""

import os
import json
import logging
import csv
from typing import Dict, List, Any, Optional

from src.utils.error_handler import ExportError, FileError, ErrorSeverity

logger = logging.getLogger(__name__)

class TranscriptExporter:
    """Class for exporting transcripts in various formats."""
    
    def __init__(self):
        """Initialize the transcript exporter."""
        logger.debug("TranscriptExporter initialized")
    
    def export_transcript(self, segments: List[Dict[str, Any]], output_path: str, 
                         format_type: str = 'txt', include_speaker: bool = True,
                         include_timestamps: bool = True) -> bool:
        """Export transcript segments to a file.
        
        Args:
            segments: List of transcript segments
            output_path: Path to save the exported file
            format_type: Export format type ('txt', 'srt', 'vtt', 'json', 'csv')
            include_speaker: Whether to include speaker information
            include_timestamps: Whether to include timestamps
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ExportError: If there's an error during export
            FileError: If there's an error with file operations
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Check if segments are valid
            if not segments:
                raise ExportError(
                    "No segments provided for export",
                    ErrorSeverity.WARNING,
                    {"output_path": output_path}
                )
            
            # Export based on format
            if format_type == 'txt':
                return self._export_transcript_txt(segments, output_path, include_speaker, include_timestamps)
            elif format_type == 'srt':
                return self._export_transcript_srt(segments, output_path)
            elif format_type == 'vtt':
                return self._export_transcript_vtt(segments, output_path)
            elif format_type == 'json':
                return self._export_transcript_json(segments, output_path)
            elif format_type == 'csv':
                return self._export_transcript_csv(segments, output_path)
            
            return False
            
        except (PermissionError, IOError) as e:
            raise FileError(
                f"File operation error: {str(e)}",
                ErrorSeverity.ERROR,
                {"output_path": output_path, "error": str(e)}
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            raise ExportError(
                f"Unexpected error during transcript export: {str(e)}",
                ErrorSeverity.ERROR,
                {"output_path": output_path, "format": format_type}
            ) from e
    
    def _export_transcript_txt(self, segments: List[Dict[str, Any]], output_path: str,
                              include_speaker: bool = True, include_timestamps: bool = True) -> bool:
        """Export transcript segments to a plain text file.
        
        Args:
            segments: List of transcript segments
            output_path: Path to save the exported file
            include_speaker: Whether to include speaker information
            include_timestamps: Whether to include timestamps
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            FileError: If there's an error with file operations
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    # Format line
                    line = ""
                    
                    # Add timestamp if requested
                    if include_timestamps:
                        start = self._format_time(segment['start'])
                        end = self._format_time(segment['end'])
                        line += f"[{start} - {end}] "
                    
                    # Add speaker if requested and available
                    if include_speaker and 'speaker' in segment and segment['speaker']:
                        speaker_name = segment.get('speaker_name', f"Speaker {segment['speaker']}")
                        line += f"{speaker_name}: "
                    
                    # Add text
                    line += segment['text']
                    
                    # Write line
                    f.write(line + "\n\n")
            
            logger.info(f"Exported transcript to text file: {output_path}")
            return True
            
        except (PermissionError, IOError) as e:
            raise FileError(
                f"Error writing to text file: {str(e)}",
                ErrorSeverity.ERROR,
                {"output_path": output_path, "error": str(e)}
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            raise ExportError(
                f"Unexpected error during text export: {str(e)}",
                ErrorSeverity.ERROR,
                {"output_path": output_path}
            ) from e
    
    def _export_transcript_srt(self, segments: List[Dict[str, Any]], output_path: str) -> bool:
        """Export transcript segments to a SubRip subtitle file.
        
        Args:
            segments: List of transcript segments
            output_path: Path to save the exported file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments):
                    # Subtitle number
                    f.write(f"{i+1}\n")
                    
                    # Timestamp
                    start = self._format_time_srt(segment['start'])
                    end = self._format_time_srt(segment['end'])
                    f.write(f"{start} --> {end}\n")
                    
                    # Text with speaker if available
                    text = segment['text']
                    if 'speaker' in segment and segment['speaker']:
                        speaker_name = segment.get('speaker_name', f"Speaker {segment['speaker']}")
                        text = f"{speaker_name}: {text}"
                    
                    f.write(f"{text}\n\n")
            
            logger.info(f"Exported transcript to SRT file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting transcript to SRT: {e}")
            return False
    
    def _export_transcript_vtt(self, segments: List[Dict[str, Any]], output_path: str) -> bool:
        """Export transcript segments to a WebVTT subtitle file.
        
        Args:
            segments: List of transcript segments
            output_path: Path to save the exported file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("WEBVTT\n\n")
                
                for i, segment in enumerate(segments):
                    # Cue identifier (optional)
                    f.write(f"cue-{i+1}\n")
                    
                    # Timestamp
                    start = self._format_time_vtt(segment['start'])
                    end = self._format_time_vtt(segment['end'])
                    f.write(f"{start} --> {end}\n")
                    
                    # Text with speaker if available
                    text = segment['text']
                    if 'speaker' in segment and segment['speaker']:
                        speaker_name = segment.get('speaker_name', f"Speaker {segment['speaker']}")
                        text = f"{speaker_name}: {text}"
                    
                    f.write(f"{text}\n\n")
            
            logger.info(f"Exported transcript to VTT file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting transcript to VTT: {e}")
            return False
    
    def _export_transcript_json(self, segments: List[Dict[str, Any]], output_path: str) -> bool:
        """Export transcript segments to a JSON file.
        
        Args:
            segments: List of transcript segments
            output_path: Path to save the exported file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a copy of segments to avoid modifying the original
            export_data = []
            
            for segment in segments:
                # Create a copy of the segment
                export_segment = segment.copy()
                
                # Add speaker name if available
                if 'speaker' in export_segment and export_segment['speaker']:
                    if 'speaker_name' not in export_segment:
                        export_segment['speaker_name'] = f"Speaker {export_segment['speaker']}"
                
                export_data.append(export_segment)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported transcript to JSON file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting transcript to JSON: {e}")
            return False
    
    def _export_transcript_csv(self, segments: List[Dict[str, Any]], output_path: str) -> bool:
        """Export transcript segments to a CSV file.
        
        Args:
            segments: List of transcript segments
            output_path: Path to save the exported file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                # Define CSV fields
                fieldnames = ['start', 'end', 'speaker', 'speaker_name', 'text']
                
                # Create CSV writer
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write segments
                for segment in segments:
                    # Create a copy of the segment with only the required fields
                    row = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'speaker': segment.get('speaker', ''),
                        'speaker_name': segment.get('speaker_name', f"Speaker {segment.get('speaker', '')}") if segment.get('speaker') else '',
                        'text': segment['text']
                    }
                    
                    writer.writerow(row)
            
            logger.info(f"Exported transcript to CSV file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting transcript to CSV: {e}")
            return False
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string (MM:SS.ms)
        """
        minutes = int(seconds // 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:06.3f}"
    
    def _format_time_srt(self, seconds: float) -> str:
        """Format time in seconds to SRT format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string (HH:MM:SS,ms)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"
    
    def _format_time_vtt(self, seconds: float) -> str:
        """Format time in seconds to WebVTT format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string (HH:MM:SS.ms)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}" 