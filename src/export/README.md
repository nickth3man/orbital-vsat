# Export Module

The export module provides functionality for exporting audio segments and transcripts in various formats.

## Features

- Export transcripts in multiple formats:
  - Plain text (TXT)
  - SubRip Subtitle (SRT)
  - WebVTT (VTT)
  - JSON
  - CSV

- Export audio in multiple formats:
  - WAV
  - MP3
  - FLAC

- Export options:
  - Complete transcript
  - Audio segments based on time range
  - Speaker-specific audio
  - Word-level audio
  - Selected words/phrases

## Usage

### ExportManager

The `ExportManager` class provides methods for exporting transcripts and audio segments:

```python
from vsat.src.export.export_manager import ExportManager

# Create export manager
export_manager = ExportManager()

# Export transcript to text file
export_manager.export_transcript(
    segments,
    output_path="transcript.txt",
    format_type="txt",
    include_speaker=True,
    include_timestamps=True
)

# Export audio segment
export_manager.export_audio_segment(
    audio_file="recording.wav",
    output_path="segment.wav",
    start=10.5,
    end=15.2,
    format_type="wav"
)

# Export speaker audio
export_manager.export_speaker_audio(
    audio_file="recording.wav",
    segments=segments,
    output_dir="speaker_audio",
    speaker_id=1,
    format_type="wav"
)

# Export word audio
export_manager.export_word_audio(
    audio_file="recording.wav",
    word=word_data,
    output_path="word.wav",
    format_type="wav",
    padding_ms=50
)

# Export selection of words
export_manager.export_selection(
    audio_file="recording.wav",
    words=selected_words,
    output_path="selection.wav",
    format_type="wav",
    include_transcript=True
)
```

## Integration with UI

The export functionality is integrated into the main application UI through:

1. Export menu options in the File menu
2. Context menu options in the transcript view
3. Export buttons in the toolbar

## Supported Formats

### Transcript Formats

- **TXT**: Plain text format with optional speaker labels and timestamps
- **SRT**: SubRip Subtitle format, commonly used for video subtitles
- **VTT**: WebVTT format, used for web video subtitles
- **JSON**: Structured format with all segment data
- **CSV**: Tabular format with segment data

### Audio Formats

- **WAV**: Uncompressed audio format
- **MP3**: Compressed audio format with good quality/size ratio
- **FLAC**: Lossless compressed audio format 