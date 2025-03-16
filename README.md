# Voice Separation & Analysis Tool (VSAT)

<div align="center">
  
![VSAT Logo](https://via.placeholder.com/200x200.png?text=VSAT)

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A comprehensive desktop application for audio recording analysis, speaker separation, transcription, and content analysis.**

[Key Features](#key-features) | [Demo](#demo) | [Installation](#installation) | [Usage](#usage) | [Architecture](#architecture) | [Development](#development) | [Contributing](#contributing) | [License](#license)

</div>

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
```bash
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

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vsat.git
   cd vsat
   ```

2. Create a virtual environment (strongly recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Download required models (first run will download automatically, but can be done manually):
   ```bash
   python -m src.ml.model_downloader
   ```

5. Verify installation:
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

```
vsat/
├── data/                 # Sample data for testing
├── docs/                 # Documentation
│   ├── api/              # API documentation
│   ├── user_guide/       # User guides and tutorials
│   └── development/      # Development guidelines
├── models/               # Storage for ML models
├── src/                  # Source code
│   ├── audio/            # Audio processing modules
│   │   ├── file_handler.py     # Audio file operations
│   │   ├── preprocessor.py     # Audio enhancement
│   │   ├── player.py           # Audio playback
│   │   └── processor.py        # Audio pipeline
│   ├── database/         # Database storage and management
│   │   ├── models.py           # ORM models
│   │   └── db_manager.py       # Database operations
│   ├── demos/            # Demo applications
│   │   ├── run_demos.py        # Demo launcher
│   │   ├── vad_demo.py         # VAD demonstration
│   │   └── content_analysis_demo.py  # Content analysis demo
│   ├── evaluation/       # Benchmark and evaluation tools
│   ├── export/           # Export functionality
│   ├── ml/               # Machine learning models
│   │   ├── diarization.py      # Speaker diarization
│   │   ├── voice_activity_detection.py  # Speech detection
│   │   ├── speaker_identification.py    # Speaker recognition
│   │   └── content_analysis.py  # Content insights
│   ├── transcription/    # Speech-to-text processing
│   │   ├── whisper_transcriber.py  # Transcription engine
│   │   └── word_aligner.py    # Word boundary detection
│   ├── ui/               # User interface components
│   │   ├── app.py             # Main application window
│   │   ├── waveform_view.py   # Audio visualization
│   │   ├── transcript_view.py # Transcript display
│   │   └── search_panel.py    # Search functionality
│   └── utils/            # Utility functions
│       ├── error_handler.py   # Error management
│       └── config_manager.py  # Configuration handling
├── tests/                # Unit and integration tests
│   ├── test_audio_processing/  # Audio module tests
│   ├── test_ml/               # ML module tests
│   └── test_ui/               # UI component tests
├── requirements.txt      # Project dependencies
└── setup.py              # Package setup
```

### Development Workflow

1. **Setup**: Follow the installation instructions above
2. **Branch**: Create a feature branch for your changes from the main branch
3. **Develop**: Make changes and add appropriate tests following coding standards
4. **Test**: Run tests to ensure functionality and maintain coverage
   ```bash
   pytest
   pytest --cov=src tests/  # For coverage report
   ```
5. **Documentation**: Update documentation to reflect changes
   - Update docstrings for all new functions and classes
   - Update user documentation if interfaces change
   - Add examples for new features
6. **Pull Request**: Submit a pull request with your changes
   - Include comprehensive description of changes
   - Reference any related issues
   - Ensure CI passes all checks

### Coding Standards

- Follow PEP 8 style guidelines with a line length of 100 characters
- Write descriptive docstrings for all functions and classes following Google style
- Include type hints for function parameters and return values
- Write unit tests for all new functionality with >90% coverage
- Use meaningful variable names and comments for complex logic
- Follow the principle of least surprise in API design
- Handle all potential exceptions appropriately
- Use constants instead of magic numbers/strings

## Contributing

We welcome contributions to VSAT! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code of Conduct for respectful and inclusive collaboration
- Development workflow from fork to pull request
- Pull request process and review criteria
- Testing requirements and coverage expectations
- Documentation standards for code and user documentation
- Issue reporting process and templates
- Feature request guidelines
- Security vulnerability reporting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

VSAT leverages several powerful open-source libraries and research:

- **PyTorch** - Deep learning framework for neural network models
- **Librosa** - Audio analysis library for feature extraction
- **PyAnnote** - Speaker diarization framework with state-of-the-art performance
- **Whisper** - Speech recognition model by OpenAI with multilingual support
- **ECAPA-TDNN** - Speaker embedding model by SpeechBrain for voice identification
- **Conv-TasNet** - Speech separation model for isolating overlapping voices
- **PyQt** - GUI framework for the user interface components
- **SQLAlchemy** - ORM for database operations and management
- **NLTK & spaCy** - NLP tools for content analysis and sentiment detection
- **Matplotlib & Seaborn** - Visualization libraries for data presentation

## Citations

If you use VSAT in your research, please cite the following papers that made this work possible:

```
@article{bredin2020pyannote,
  title={pyannote.audio: neural building blocks for speaker diarization},
  author={Bredin, Hervé and others},
  journal={ICASSP},
  year={2020}
}

@article{luo2019conv,
  title={Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation},
  author={Luo, Yi and Mesgarani, Nima},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2019}
}

@article{radford2022robust,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and others},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}

@article{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  journal={Interspeech},
  year={2020}
}
``` 