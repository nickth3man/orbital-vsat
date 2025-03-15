# Voice Separation & Analysis Tool (VSAT)

VSAT is a Windows desktop application designed to process audio recordings of conversations, separate individual speakers, identify them, create a searchable database of speakers with their voice clips, and generate transcripts for each segment.

## Features

- **Audio Processing & Speaker Separation**: Load and process audio files, enhance audio quality, separate overlapping voices, and identify speech segments.
- **Speaker Diarization & Identification**: Determine who spoke when, create speaker profiles, match speakers across recordings, and analyze speaking patterns.
- **Transcription & Word-level Processing**: Generate accurate transcripts with timestamps, enable precise word boundary detection, and analyze transcript content.
- **Database & Organization**: Store processed data in a searchable database, search by transcript content, speaker, date, and metadata, and manage database size and performance.
- **Export & Sharing**: Export clean audio files for each speaker, extract audio based on word selections from transcript, and export transcripts in various formats.
- **User Interface**: Interactive visualization of audio with speaker coloring, comprehensive UI controls, and guided workflows for common tasks.

## Installation

### Requirements

- Python 3.11 or higher
- Windows 10 or higher
- Recommended hardware: i7/Ryzen 7 CPU, 32GB RAM, NVIDIA GPU with 8GB+ VRAM

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/vsat.git
   cd vsat
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Usage

### Basic Operation

1. Launch the application:
   ```
   vsat
   ```

2. Import an audio file using the File menu or drag-and-drop.

3. Process the audio to separate speakers and generate transcriptions.

4. Use the interactive timeline to explore the conversation.

5. Export individual speaker audio or specific word segments as needed.

### Advanced Features

- **Speaker Management**: Name speakers and build a database of known voices.
- **Search Functionality**: Find specific content across all processed recordings.
- **Custom Export**: Export selected segments in various formats.

## Development

The project is structured as follows:

- `src/audio`: Audio file handling, preprocessing, and speaker separation
- `src/database`: Database schema and operations
- `src/ml`: Machine learning model interfaces
- `src/transcription`: Speech-to-text and word-level processing
- `src/ui`: User interface components
- `tests`: Unit and integration tests
- `docs`: Documentation
- `models`: Storage for ML models
- `data`: Sample data for testing

## License

[Specify your license here]

## Acknowledgments

- [List any libraries, tools, or resources used in the project] 