# Voice Separation & Analysis Tool (VSAT)

![VSAT Logo](https://via.placeholder.com/200x200.png?text=VSAT)

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A comprehensive desktop application for audio recording analysis, speaker separation, transcription, and content analysis.**

[Key Features](#key-features) | [Demo](#demo) | [Installation](#installation) | [Usage](#usage) | [Step-by-Step Guide](#step-by-step-guide) | [Architecture](#architecture) | [Development](#development) | [Contributing](#contributing) | [License](#license)

## Overview

VSAT (Voice Separation & Analysis Tool) is a sophisticated Windows desktop application built for professionals who need to process, analyze, and extract insights from conversation recordings. The tool combines state-of-the-art machine learning models with an intuitive user interface to deliver a complete solution for audio analysis.

By leveraging recent advancements in speaker diarization and speech recognition technologies, VSAT can:

- Separate individual speakers from complex overlapping conversations
- Generate accurate transcripts with word-level timestamps
- Create searchable speaker profiles that work across multiple recordings
- Analyze content including sentiment, topics, and important moments
- Export processed data in multiple formats for further use
- Provide detailed analytics on speaker patterns and conversation dynamics

## Key Features

### Audio Processing & Speaker Separation

- **Audio File Handling**: Support for WAV, MP3, FLAC, and other common audio formats with automatic format detection
- **Audio Enhancement**: Advanced noise reduction, normalization, and equalization of audio for improved quality
- **Voice Activity Detection (VAD)**: Accurate identification of speech segments using both energy-based and ML-based approaches with >95% accuracy
- **Speaker Separation**: State-of-the-art deep learning models (leveraging Conv-TasNet architecture) to isolate individual speakers in overlapping speech
- **Audio Preprocessing**: Customizable preprocessing profiles for different recording environments (conference room, interview, noisy background)
- **Chunked Processing**: Efficient processing of long recordings through automatic chunking with overlap handling

### Speaker Diarization & Identification

- **Speaker Diarization**: Advanced diarization using PyAnnote's SOTA neural models to determine who spoke when with <15% DER for clear recordings
- **Speaker Identification**: ECAPA-TDNN voice embedding model to create and match speaker profiles with >85% accuracy across recordings
- **Voice Print Database**: Persistent storage of speaker characteristics for cross-recording identification and voice profile management
- **Speaker Statistics**: Comprehensive analysis of speaking patterns, duration, turn-taking, and interaction dynamics
- **Manual Speaker Labeling**: Intuitive tools for manually correcting and labeling speakers with continuous learning from corrections

### Transcription & Content Analysis

- **Automatic Transcription**: High-accuracy speech-to-text using the Whisper model with support for multiple languages and <15% WER for clear speech
- **Word-Level Alignment**: Precise boundary detection for each word in the transcript with <100ms precision
- **Text Analysis**:
  - **Topic Modeling**: Extract main topics from conversations using LDA and NMF algorithms with visualization
  - **Keyword Extraction**: Identify important terms and phrases with frequency and relevance scoring
  - **Summarization**: Generate concise summaries of long recordings with customizable length
  - **Sentiment Analysis**: Real-time analysis of emotional tone throughout the conversation with visualization of sentiment changes
  - **Important Moment Detection**: Automatic highlighting of significant sections based on content, speaker changes, and sentiment shifts

### Database & Organization

- **SQLite Database**: Efficient local storage of all processed data with optimized schema design
- **Full-Text Search**: Powerful search functionality for transcripts by content, speaker, timestamp, and metadata
- **Recording Management**: Comprehensive organization tools for large collections of recordings including tagging and filtering
- **Backup & Archive**: Automated tools for backing up data and archiving old recordings with integrity verification
- **Performance Optimization**: Advanced database optimization for handling large volumes of data with minimal performance impact

### Export & Sharing

- **Audio Export**: Export clean audio for individual speakers, segments, or specific words with format selection
- **Transcript Export**: Multiple formats including TXT, SRT, JSON, and CSV with customizable metadata inclusion
- **Batch Processing**: Powerful tools for processing and exporting multiple recordings simultaneously
- **Custom Selections**: Precision export of specific portions based on speaker, time range, or content search
- **Error Handling**: Robust error tracking and recovery during export operations with detailed logging and retry mechanisms

### User Interface

- **Waveform Visualization**: Interactive, high-resolution waveform display with color-coded speaker segments
- **Timeline Navigation**: Intuitive timeline with markers for speech segments, speaker changes, and important moments
- **Workflow Guidance**: Guided step-by-step process for common tasks with help tooltips
- **Transcript Editor**: Comprehensive tools for manually correcting transcription errors with spell-checking and formatting
- **Accessibility Features**: Full support for screen readers, keyboard navigation, UI scaling, and high-contrast modes

## Demo

VSAT includes several interactive demo applications to showcase its capabilities:

### Voice Activity Detection Demo

![VAD Demo](https://via.placeholder.com/800x400.png?text=VAD+Demo)

Demonstrates the Voice Activity Detection capabilities with:

- Interactive visualization of speech segments with real-time updates
- Adjustable sensitivity settings with immediate visual feedback
- Advanced configuration options for different audio environments
- Export functionality for detected segments with timestamp metadata
- Live audio input processing option for real-time demonstrations

### Content Analysis Demo

![Content Analysis Demo](https://via.placeholder.com/800x400.png?text=Content+Analysis+Demo)

Showcases the content analysis features:

- Topic extraction with interactive topic modeling visualization
- Keyword identification with relevance scoring and highlighting
- Sentiment analysis visualization showing emotional flow over time
- Important moment detection with reasoning and context
- Summary generation with adjustable length and focus options

Run the demos using:

```python
python -m src.demos.run_demos
```

The demo launcher provides a user-friendly interface to select and run any of the available demonstrations.

## Installation

### Requirements

- **Python**: 3.11 or higher
- **Operating System**: Windows 10 or higher
- **Hardware Recommendations**:
  - CPU: i7/Ryzen 7 or better
  - RAM: 32GB (minimum 16GB)
  - GPU: NVIDIA GPU with 8GB+ VRAM (for optimal performance)
  - Storage: SSD with at least 10GB free space for application and models

### Dependencies

VSAT relies on several key libraries:

- **PyTorch & TorchAudio**: For deep learning models and audio processing
- **Librosa**: For audio feature extraction and advanced signal processing
- **PyAudio**: For real-time audio playback and recording capabilities
- **SpeechBrain**: For ECAPA-TDNN speaker embeddings and voice print generation
- **PyAnnote**: For state-of-the-art speaker diarization with neural models
- **Faster-Whisper**: For optimized speech transcription with word-level timestamps
- **PyQt6**: For the responsive graphical user interface components
- **SQLAlchemy**: For robust database interactions and schema management
- **NLTK & scikit-learn**: For NLP tasks including sentiment analysis and topic modeling
- **Matplotlib & Seaborn**: For visualization of audio data and analysis results

### Setup

1. Create a virtual environment (strongly recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

2. Install the package in development mode:

   ```bash
   pip install -e .
   ```

3. Download required models (first run will download automatically, but can be done manually):

   ```bash
   python -m src.ml.model_downloader
   ```

4. Verify installation:

   ```bash
   python -m src.utils.system_check
   ```

## Usage

### Basic Operation

1. Launch the application:

   ```bash
   python -m src.main
   ```

   Or if installed as a package:

   ```bash
   vsat
   ```

2. Import an audio file:
   - Use the File → Open menu
   - Drag and drop a file into the application window
   - Use recent files list for quick access
   - Connect to external recording device for live input

3. Process the audio:
   - Configure processing options in the Processing panel
   - Select preprocessing profile based on recording environment
   - Set speaker detection sensitivity and options
   - Click "Process Audio" to start the analysis pipeline
   - Monitor detailed progress in the status bar with cancelation option

4. Explore the results:
   - Navigate the audio using the interactive waveform view
   - Review the transcript in the transcript panel with speaker highlighting
   - Explore speaker segments with color coding and filtering options
   - View content analysis in the analysis tab with visualization options
   - Search for specific content across the entire recording

5. Export results:
   - Use Export menu to access comprehensive export options
   - Select specific segments, speakers, or entire recording
   - Choose from multiple output formats with customization
   - Set batch export parameters for multiple files
   - Monitor export progress with detailed status updates

### Advanced Workflows

#### Speaker Profile Management

1. Access the Speaker Management dialog from the Tools menu
2. Create, edit, or delete speaker profiles with custom metadata
3. Train the system with additional voice samples for improved accuracy
4. Link speakers across multiple recordings for consistent identification
5. Export and import speaker profiles for backup or sharing

#### Content Search and Analysis

1. Use the Search panel to find specific content with advanced query options
2. Filter by speaker, time range, sentiment, or metadata
3. View and export search results with surrounding context
4. Use the Analysis tab to explore content insights with interactive visualizations
5. Generate custom reports based on search results and analysis

#### Batch Processing

1. Select multiple files in the File → Batch Process menu
2. Configure processing settings with templates for common scenarios
3. Set processing priority and resource allocation
4. Run batch processing with detailed progress monitoring
5. Review comprehensive results in the batch summary view
6. Export batch processing reports in multiple formats

## Step-by-Step Guide

This comprehensive guide walks you through every aspect of using VSAT, from installation to advanced features.

### 1. Installation and First Launch

#### Complete Installation

1. **Install Python 3.11+**:
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation by opening Command Prompt and typing:

     ```bash
     python --version
     ```

2. **Install Required Libraries**:
   - Open Command Prompt as Administrator
   - Navigate to your desired installation directory
   - Clone the repository:

     ```bash
     git clone https://github.com/yourusername/vsat.git
     cd vsat
     ```

   - Create and activate a virtual environment:

     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   - Install VSAT and dependencies:

     ```bash
     pip install -e .
     ```

3. **First Launch**:
   - From the activated virtual environment, run:

     ```bash
     python -m src.main
     ```

   - The first launch will download required ML models (approximately 2-4GB)
   - A progress bar will display download status
   - After model download completes, the main application window will appear

#### Troubleshooting First Launch

- **Missing Dependencies**: If you encounter errors about missing packages, run:

  ```bash
  pip install -r requirements.txt
  ```

- **CUDA Issues**: If you have an NVIDIA GPU but encounter CUDA errors:
  1. Verify your GPU drivers are up to date
  2. Install the appropriate CUDA toolkit version for your PyTorch installation
  3. Alternatively, force CPU mode by editing `~/.vsat/config.yaml` and setting `device: cpu`

- **Model Download Failures**: If model downloads fail:
  1. Check your internet connection
  2. Try running with administrator privileges
  3. Manually download models using:

     ```bash
     python -m src.ml.model_downloader --force
     ```

### 2. Loading and Processing Audio Files

#### Supported Audio Formats

VSAT supports the following audio formats:

- WAV (`.wav`) - Recommended for highest quality
- MP3 (`.mp3`) - Good for compressed files
- FLAC (`.flac`) - Lossless compression
- OGG (`.ogg`) - Open format alternative
- M4A (`.m4a`) - Common for recordings from mobile devices

#### Loading Audio Files

1. **From the File Menu**:
   - Click `File → Open` in the top menu
   - Navigate to your audio file
   - Select the file and click "Open"

2. **Using Drag and Drop**:
   - Locate your audio file in File Explorer
   - Drag the file directly into the VSAT application window
   - Release to begin loading

3. **From Recent Files**:
   - Click `File → Recent Files`
   - Select from the list of recently processed files

#### Processing Options

Before processing begins, you can configure several options:

1. **Preprocessing Profile**:
   - **Default**: Balanced settings suitable for most recordings
   - **Conference Room**: Optimized for multiple speakers in good acoustic environments
   - **Interview**: Tuned for two-person conversations
   - **Noisy Background**: Enhanced noise reduction for recordings with background noise
   - **Custom**: Create your own profile with manual settings

2. **Speaker Detection Settings**:
   - **Sensitivity**: Adjust how aggressively the system detects speaker changes
     - Low: Fewer speaker changes, may miss some transitions
     - Medium: Balanced approach (recommended)
     - High: More sensitive, may create extra speaker segments
   - **Expected Speakers**: Set the expected number of speakers (2-6)
   - **Min Segment Duration**: Set minimum duration for speaker segments (0.5-2.0 seconds)

3. **Processing Mode**:
   - **Standard**: Process the entire file at once (for files under 30 minutes)
   - **Chunked**: Process in segments (recommended for files over 30 minutes)
   - **High Accuracy**: Use larger models for better results but slower processing

4. **Advanced Options**:
   - **Language**: Specify the primary language for better transcription
   - **GPU Acceleration**: Enable/disable GPU usage
   - **Cache Results**: Store intermediate results for faster reprocessing

#### Processing Workflow

1. After configuring options, click "Process Audio"
2. The progress bar will show:
   - Audio loading and validation
   - Preprocessing (noise reduction, normalization)
   - Voice activity detection
   - Speaker diarization
   - Transcription
   - Speaker identification
   - Content analysis
3. For large files, estimated time remaining will be displayed
4. Processing can be canceled at any time by clicking "Cancel"
5. Upon completion, the interface will update with results

### 3. Navigating the User Interface

#### Main Window Layout

The VSAT interface is divided into several key areas:

1. **Menu Bar**: Access to all functions and features
2. **Toolbar**: Quick access to common actions
3. **Waveform View**: Visual representation of the audio with speaker coloring
4. **Transcript Panel**: Text transcript with speaker labels
5. **Tools Panel**: Contains search, analysis, and other tools
6. **Status Bar**: Displays current status and progress information

#### Waveform View

The waveform view provides a visual representation of your audio:

1. **Navigation**:
   - **Zoom**: Use mouse wheel or zoom slider
   - **Pan**: Click and drag the waveform horizontally
   - **Position**: Click anywhere to set playback position

2. **Speaker Segments**:
   - Each speaker is color-coded
   - Hover over a segment to see speaker name and confidence score
   - Right-click a segment for additional options

3. **Selection**:
   - Click and drag to select a range
   - Double-click a speaker segment to select it entirely
   - Selected ranges can be played, exported, or analyzed

4. **Playback Controls**:
   - Play/Pause: Space bar or play button
   - Stop: Esc key or stop button
   - Jump Forward/Back: Arrow keys
   - Adjust Volume: Volume slider or up/down arrow keys

#### Transcript Panel

The transcript panel shows the text transcript with speaker information:

1. **Navigation**:
   - Scroll to browse the transcript
   - Click any word to jump to that position in the audio
   - Search using Ctrl+F to find specific text

2. **Speaker Labels**:
   - Each speaker has a consistent color matching the waveform
   - Speaker names can be edited by right-clicking
   - Confidence scores are shown for each segment

3. **Editing**:
   - Double-click text to edit transcription
   - Right-click a segment to change speaker assignment
   - Use the Edit menu for additional editing options

4. **Selection**:
   - Click and drag to select words or phrases
   - Double-click to select a word
   - Triple-click to select an entire segment
   - Selected text can be copied, exported, or analyzed

#### Tools Panel

The tools panel contains tabs for different functions:

1. **Search Tab**:
   - Full-text search across the transcript
   - Filter by speaker, time range, or confidence
   - Results show context and can be clicked to navigate
   - Advanced options include regex, case sensitivity, and whole word matching

2. **Content Analysis Tab**:
   - Topic modeling visualization
   - Keyword extraction with frequency analysis
   - Sentiment analysis timeline
   - Important moment detection
   - Summary generation

3. **Speaker Tab**:
   - List of identified speakers
   - Speaking statistics for each speaker
   - Voice print management
   - Speaker comparison tools

### 4. Working with Speaker Profiles

#### Understanding Speaker Identification

VSAT uses voice prints (mathematical representations of voice characteristics) to identify speakers:

1. **Initial Identification**: During processing, VSAT:
   - Separates the audio into speaker segments
   - Creates voice prints for each speaker
   - Compares with existing voice prints in the database
   - Either matches to known speakers or creates new profiles

2. **Voice Print Quality**:
   - Longer speech segments produce better voice prints
   - Clear audio without background noise improves accuracy
   - Multiple recordings of the same speaker increase recognition accuracy

#### Managing Speaker Profiles

Access the Speaker Management dialog from `Tools → Speaker Management`:

1. **Viewing Speakers**:
   - All speakers in the database are listed
   - Each entry shows name, recording count, and total duration
   - Preview audio samples for each speaker

2. **Creating New Speakers**:
   - Click "New Speaker"
   - Enter a name and optional metadata
   - Add voice samples by recording or importing audio files
   - Save to create the profile

3. **Editing Speakers**:
   - Select a speaker and click "Edit"
   - Rename or update metadata
   - Add additional voice samples to improve recognition
   - Merge duplicate profiles if needed

4. **Training the System**:
   - Select a speaker
   - Click "Add Samples"
   - Choose from existing recordings or import new audio
   - Select clear speech segments for the speaker
   - Add to the voice print database

5. **Cross-Recording Identification**:
   - Enable "Link Speakers Across Recordings" in settings
   - Set the similarity threshold (0.75 recommended)
   - Process new recordings to automatically identify known speakers

#### Correcting Speaker Assignments

To correct misidentified speakers in the current recording:

1. **Manual Reassignment**:
   - Right-click a segment in the waveform or transcript
   - Select "Change Speaker"
   - Choose the correct speaker or create a new one
   - Apply to this segment only or all similar segments

2. **Bulk Reassignment**:
   - Select multiple segments (Ctrl+click or selection tool)
   - Right-click and select "Assign to Speaker"
   - Choose the correct speaker
   - Click "Apply" to update all selected segments

3. **Learning from Corrections**:
   - After making corrections, click "Update Voice Prints"
   - The system will incorporate your corrections into the speaker models
   - Future processing will use the improved voice prints

### 5. Searching and Analyzing Content

#### Basic Search

The Search panel provides powerful text search capabilities:

1. **Simple Search**:
   - Enter text in the search box
   - Press Enter or click the search icon
   - Results appear in the results list with context

2. **Search Options**:
   - **Case Sensitive**: Match exact capitalization
   - **Whole Word**: Match complete words only
   - **Regular Expression**: Use regex patterns for complex searches
   - **Context Size**: Adjust how much text is shown around matches

3. **Filtering Results**:
   - By Speaker: Show only results from specific speakers
   - By Time Range: Limit search to a portion of the recording
   - By Confidence: Filter by transcription confidence level

4. **Navigating Results**:
   - Click any result to jump to that position
   - Use Up/Down arrows to move through results
   - Press F3 to move to the next result

#### Advanced Content Analysis

The Content Analysis tab provides deeper insights:

1. **Topic Modeling**:
   - Click "Generate Topics" to extract main conversation topics
   - Adjust the number of topics (3-10 recommended)
   - View topic distribution across the recording
   - See key terms associated with each topic

2. **Keyword Extraction**:
   - View automatically extracted keywords and phrases
   - Sort by frequency, relevance, or position
   - Filter by speaker or segment
   - Export keyword list with timestamps

3. **Sentiment Analysis**:
   - View sentiment timeline showing emotional tone
   - Positive/negative/neutral classification
   - Speaker-specific sentiment tracking
   - Identify emotional shifts in the conversation

4. **Important Moment Detection**:
   - System automatically identifies significant moments:
     - Topic changes
     - Emotional shifts
     - Key information points
     - Areas of agreement/disagreement
   - Click any moment to jump to that position
   - Add manual markers for your own important points

5. **Summary Generation**:
   - Click "Generate Summary"
   - Select summary length (short, medium, long)
   - Choose focus areas (topics, speakers, key points)
   - View and export the generated summary

### 6. Exporting Results

#### Exporting Transcripts

VSAT supports multiple transcript export formats:

1. **Text Format Options**:
   - **Plain Text**: Simple text file with optional timestamps
   - **SRT/VTT**: Subtitle formats with timing information
   - **JSON**: Structured data with all metadata
   - **CSV**: Tabular format for spreadsheet analysis
   - **DOCX**: Formatted Word document with speaker labels

2. **Export Process**:
   - Select `Export → Export Transcript` from the menu
   - Choose the format
   - Configure format-specific options:
     - Include speaker names
     - Add timestamps
     - Include confidence scores
     - Format options (colors, fonts, etc.)
   - Select export location
   - Click "Export"

3. **Partial Transcript Export**:
   - Select specific segments in the transcript or waveform
   - Right-click and select "Export Selection"
   - Follow the same process as full export

#### Exporting Audio

Export processed audio in various formats:

1. **Full Recording Export**:
   - Select `Export → Export Audio` from the menu
   - Choose format (WAV, MP3, FLAC)
   - Set quality options
   - Select export location
   - Click "Export"

2. **Speaker Separation Export**:
   - Select `Export → Export Speakers` from the menu
   - Choose which speakers to export
   - Select format and quality
   - Enable "Enhance Separation" for improved quality
   - Click "Export" to create individual files for each speaker

3. **Segment Export**:
   - Select segments in the waveform or transcript
   - Right-click and select "Export Audio Selection"
   - Configure format and options
   - Click "Export"

4. **Word-Level Export**:
   - Select specific words in the transcript
   - Right-click and select "Export Word Audio"
   - Choose to export as individual files or a combined file
   - Set padding (time before/after the word)
   - Click "Export"

#### Batch Exporting

For processing multiple files:

1. **Batch Export Setup**:
   - Select `File → Batch Processing` from the menu
   - Add files using "Add Files" or drag and drop
   - Configure processing options
   - Set export options for transcripts and audio

2. **Export Templates**:
   - Save common export configurations as templates
   - Apply templates to new batch jobs
   - Schedule batch exports for off-hours processing

3. **Monitoring Progress**:
   - View detailed progress for each file
   - Estimated time remaining for the batch
   - Success/failure status for each operation
   - Error logs for troubleshooting

### 7. Customizing VSAT

#### Application Settings

Access settings via `Tools → Settings`:

1. **General Settings**:
   - Default directories for input/output
   - Recent files list size
   - Automatic updates
   - Crash reporting

2. **Processing Settings**:
   - Default processing profile
   - Model selection (small/medium/large)
   - GPU/CPU utilization
   - Memory usage limits

3. **Audio Settings**:
   - Playback device selection
   - Default volume
   - Waveform visualization options
   - Audio enhancement settings

4. **Display Settings**:
   - Color scheme for UI
   - Speaker color palette
   - Font size and family
   - Timeline zoom defaults

5. **Advanced Settings**:
   - Database location
   - Cache directory and size limits
   - Logging verbosity
   - Performance tuning options

#### Keyboard Shortcuts

VSAT offers customizable keyboard shortcuts:

1. **Default Shortcuts**:
   - **Ctrl+O**: Open file
   - **Ctrl+S**: Save project
   - **Ctrl+E**: Export
   - **Space**: Play/Pause
   - **Esc**: Stop playback
   - **Ctrl+F**: Search
   - **F3**: Next search result
   - **Ctrl+B**: Batch processing
   - **Ctrl+Z**: Undo
   - **Ctrl+Y**: Redo

2. **Customizing Shortcuts**:
   - Go to `Tools → Keyboard Shortcuts`
   - Select the action to customize
   - Press the desired key combination
   - Click "Apply" to save changes

#### Accessibility Features

VSAT includes comprehensive accessibility options:

1. **Screen Reader Support**:
   - Compatible with NVDA and JAWS
   - All UI elements have proper labels
   - Keyboard navigation for all functions

2. **Visual Adjustments**:
   - High contrast mode
   - Adjustable font size
   - Color filters for color vision deficiencies
   - Focus highlighting

3. **Keyboard Navigation**:
   - Tab order optimization
   - Focus indicators
   - Keyboard shortcuts for all functions
   - No mouse required for operation

4. **Accessibility Settings**:
   - Access via `Tools → Accessibility Settings`
   - Enable/disable specific features
   - Configure screen reader compatibility
   - Set up alternative input methods

### 8. Troubleshooting and Support

#### Common Issues and Solutions

1. **Slow Processing**:
   - Enable GPU acceleration if available
   - Use chunked processing for large files
   - Close other resource-intensive applications
   - Use smaller models for faster processing
   - Ensure your system meets recommended specifications

2. **Inaccurate Speaker Identification**:
   - Provide clearer audio samples for voice prints
   - Manually correct some segments and update voice prints
   - Adjust the similarity threshold in settings
   - Use recordings with less background noise
   - Ensure each speaker has sufficient speaking time

3. **Transcription Errors**:
   - Use a larger Whisper model for better accuracy
   - Specify the correct language in processing options
   - Improve audio quality with preprocessing
   - Manually correct errors and use "Learn from Corrections"
   - For domain-specific terminology, add custom vocabulary

4. **Application Crashes**:
   - Check log files in `~/.vsat/logs/`
   - Ensure all dependencies are correctly installed
   - Update to the latest version
   - Try processing in smaller chunks
   - Reduce model size if memory is limited

#### Getting Help

1. **Documentation**:
   - Access built-in help via `Help → Documentation`
   - Check the online documentation at [docs.vsat.example.com](https://docs.vsat.example.com)
   - Review the FAQ section for common questions

2. **Log Files**:
   - Log files are stored in `~/.vsat/logs/`
   - Include relevant logs when reporting issues
   - Enable detailed logging in settings for troubleshooting

3. **Community Support**:
   - Join the user forum at [forum.vsat.example.com](https://forum.vsat.example.com)
   - Check GitHub issues for known problems
   - Share your solutions to help others

4. **Reporting Bugs**:
   - Use `Help → Report Issue` to submit bug reports
   - Include steps to reproduce the issue
   - Attach sample files if possible (or links to files)
   - Share system information and log files

## Architecture

VSAT follows a modular architecture organized into several key components, designed for maintainability, extensibility, and performance:

### Core Components

- **Audio Module**: Handles audio file I/O, preprocessing, playback, and format conversion
- **ML Module**: Encapsulates machine learning models for voice detection, separation, diarization, and content analysis
- **Transcription Module**: Manages speech-to-text conversion, word alignment, and transcript editing
- **Database Module**: Provides robust data persistence, query optimization, and migration support
- **Export Module**: Handles export of processed data in various formats with error recovery
- **UI Module**: Implements the graphical user interface with responsive design patterns

### Processing Pipeline

1. **Audio Loading**: Load and decode the audio file with format detection and validation
2. **Preprocessing**: Apply noise reduction, normalization, and enhancement based on profile
3. **Voice Activity Detection**: Identify speech segments with configurable sensitivity
4. **Speaker Diarization**: Determine speaker boundaries and segment the audio
5. **Speech Separation**: Isolate individual speakers using deep learning models
6. **Transcription**: Generate text from speech with language detection
7. **Word Alignment**: Align words with precise timestamps for navigation
8. **Speaker Identification**: Match speakers to known profiles using voice prints
9. **Content Analysis**: Analyze transcript for insights, topics, and sentiment
10. **Storage**: Save processed data to database with transaction safety

### Error Handling

VSAT implements a comprehensive error handling framework designed for robustness and user-friendliness:

- **Structured exception hierarchy**: Type-specific exception classes for different categories:
  - `VSATError`: Base exception class for all VSAT-specific errors
  - `FileError`: File-related errors (not found, permission denied, etc.)
  - `AudioError`: Audio processing errors (corrupt file, unsupported format)
  - `ProcessingError`: Audio processing pipeline errors
  - `DatabaseError`: Database-related errors (connection, query, schema)
  - `ExportError`: Export-related errors (disk space, permissions)
  - `UIError`: UI-related errors

- **Severity levels**: Different levels of severity for proper handling:
  - `INFO`: Informational message, not an actual error
  - `WARNING`: Non-critical issue that doesn't stop operation
  - `ERROR`: Serious issue that prevents an operation from completing
  - `CRITICAL`: Critical error that might cause the application to terminate

- **Detailed error context**: Comprehensive information about error conditions:
  - File paths, formats, and permissions
  - Processing stage and parameters
  - System resource states
  - Stack traces for debugging

- **User-friendly error messages**: Clear and informative error dialogs with:
  - Concise error descriptions
  - Suggested solutions
  - Options to retry, cancel, or get help
  - Details expansion for technical users

- **Global exception handling**: Catch-all for unexpected exceptions with:
  - Application recovery mechanisms
  - State preservation
  - Detailed crash reports

- **Logging integration**: Comprehensive logging of all errors with:
  - Rotating log files with size management
  - Configurable verbosity levels
  - Structured log format for analysis
  - Option to submit anonymous error reports

## Development

### Project Structure

```text
vsat/
├── data/                 # Sample data for testing
├── docs/                 # Documentation
│   ├── api/              # API documentation
│   ├── user_guide/       # User guides and tutorials
│   └── development/      # Development guidelines
├── models/               # Storage for ML models
├── src/                  # Source code
│   ├── audio/            # Audio processing modules
│   ├── ml/               # Machine learning models
│   ├── transcription/    # Speech-to-text and word alignment
│   ├── database/         # Database interactions and schema
│   ├── export/           # Export functionality
│   ├── ui/               # Graphical user interface
│   └── utils/            # Utility functions
├── tests/                # Unit tests and integration tests
└── venv/                 # Virtual environment
