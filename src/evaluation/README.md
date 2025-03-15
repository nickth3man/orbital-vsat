# VSAT Evaluation and Benchmarking Module

## Overview

The Evaluation and Benchmarking module provides a comprehensive framework for assessing the performance of VSAT's key components: transcription, speaker diarization, and audio source separation. This module allows users to:

1. Calculate standard performance metrics
2. Run systematic benchmarks against standard datasets
3. Prepare and manage benchmark datasets
4. Visualize benchmark results with detailed graphs and statistics

## Module Components

### Evaluation Metrics

The module implements three key metrics for evaluating different aspects of VSAT:

#### Word Error Rate (WER)

The `WordErrorRate` class (`wer.py`) calculates the accuracy of speech transcription by comparing reference and hypothesis transcripts. It considers different types of errors:

- Substitutions: Wrong words
- Deletions: Missing words
- Insertions: Extra words

WER is calculated as: (Substitutions + Deletions + Insertions) / Total Reference Words × 100%

Lower WER values indicate better transcription accuracy.

#### Diarization Error Rate (DER)

The `DiarizationErrorRate` class (`der.py`) measures the accuracy of speaker diarization by comparing reference and hypothesis speaker segments. It accounts for three types of errors:

- Miss: Speaker segments that exist in the reference but not in the hypothesis
- False Alarm: Speaker segments that exist in the hypothesis but not in the reference
- Speaker Error: Speaker segments that are assigned to the wrong speaker

DER is calculated as: (Miss + False Alarm + Speaker Error) / Total Speech Duration × 100%

Lower DER values indicate better diarization accuracy.

#### Signal-to-Distortion Ratio (SDR)

The `SignalDistortionRatio` class (`sdr.py`) evaluates the quality of audio source separation by comparing reference and separated audio sources. It measures:

- Signal strength relative to distortion
- Quality of separation across all sources

SDR is measured in decibels (dB), with higher values indicating better separation quality.

### Benchmarking System

The benchmarking system (`benchmark.py`) provides tools for running systematic evaluations:

- `Benchmark` class for running evaluation pipelines
- Support for benchmarking transcription, diarization, and separation
- Result aggregation and detailed statistics
- JSON and CSV export for further analysis

### Dataset Preparation

The dataset preparation tools (`prepare_datasets.py`) facilitate the downloading and preparation of standard evaluation datasets:

- LibriSpeech for transcription evaluation
- AMI Corpus for diarization evaluation
- LibriMix for source separation evaluation

These tools handle downloading, extraction, and formatting of datasets for compatibility with the benchmarking system.

### Visualization Tools

The visualization tools (`visualize_results.py`) generate detailed graphs and statistics from benchmark results:

- Error distribution histograms
- Per-file performance charts
- Error component analysis
- Correlation plots (e.g., error vs. file length)
- CSV summaries for detailed analysis

## Usage

### Command-line Interface

The module provides command-line interfaces for all major functions:

#### Preparing Datasets

```bash
python -m src.evaluation.prepare_datasets --dataset librispeech-test-clean --output datasets
```

To list available datasets:

```bash
python -m src.evaluation.prepare_datasets --list
```

#### Running Benchmarks

```bash
python -m src.evaluation.run_benchmark --type transcription --dataset datasets/librispeech-test-clean --output benchmark_results
```

Options:
- `--type`: Type of benchmark to run (`transcription`, `diarization`, or `separation`)
- `--dataset`: Path to prepared dataset
- `--output`: Output directory for benchmark results
- `--model-size`: Model size for transcription benchmarks (`tiny`, `base`, `small`, `medium`, `large`)
- `--collar`: Collar size for diarization evaluation (default: 0.25 seconds)
- `--device`: Device to run inference on (`cpu`, `cuda`, or `mps`)

#### Visualizing Results

```bash
python -m src.evaluation.visualize_results --results benchmark_results/transcription_results.json --output benchmark_results/visualizations
```

### Using in Python Code

#### Calculating WER

```python
from src.evaluation.wer import WordErrorRate

wer_calculator = WordErrorRate()
wer = wer_calculator.calculate("reference transcript", "hypothesis transcript")
print(f"WER: {wer}%")
```

#### Calculating DER

```python
from src.evaluation.der import DiarizationErrorRate

der_calculator = DiarizationErrorRate()
reference_segments = [
    {"start": 0.0, "end": 2.5, "speaker": "speaker1"},
    {"start": 2.5, "end": 5.0, "speaker": "speaker2"}
]
hypothesis_segments = [
    {"start": 0.1, "end": 2.4, "speaker": "speaker1"},
    {"start": 2.6, "end": 5.0, "speaker": "speaker2"}
]
der = der_calculator.calculate(reference_segments, hypothesis_segments, collar=0.25)
print(f"DER: {der}%")
```

#### Calculating SDR

```python
from src.evaluation.sdr import SignalDistortionRatio
import numpy as np

sdr_calculator = SignalDistortionRatio()
# Create example signals (in real use, these would be loaded from audio files)
reference_source = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 48000))
estimated_source = 0.9 * reference_source + 0.1 * np.random.randn(48000)

sdr = sdr_calculator.calculate(reference_source, estimated_source, sample_rate=16000)
print(f"SDR: {sdr} dB")
```

#### Running a Benchmark

```python
from src.evaluation.benchmark import Benchmark

benchmark = Benchmark(output_dir="benchmark_results")
results = benchmark.benchmark_transcription(
    dataset_path="datasets/librispeech-test-clean",
    model_size="base",
    device="cpu"
)
print(f"Average WER: {results['average_wer']}%")
```

## Adding New Benchmark Datasets

To add a new benchmark dataset, extend the `DATASETS` dictionary in `prepare_datasets.py`:

```python
DATASETS = {
    # ... existing datasets ...
    "new-dataset-name": {
        "url": "https://example.com/dataset.zip",
        "type": "transcription",  # or "diarization" or "separation"
        "description": "Description of the dataset"
    }
}
```

Then implement a preparation function for the new dataset type if needed.

## Extending the Benchmarking System

To add a new type of benchmark, extend the `Benchmark` class in `benchmark.py`:

1. Add a new method for the benchmark type (e.g., `benchmark_new_feature`)
2. Implement the dataset loading function
3. Add the benchmarking logic
4. Implement result aggregation and saving

## Dependencies

The evaluation and benchmarking module relies on the following packages:

- numpy: For numerical operations
- scipy: For signal processing and file handling
- Levenshtein: For WER calculation
- pandas: For data manipulation and CSV export
- matplotlib and seaborn: For visualization
- mir_eval: For advanced audio evaluation metrics
- scikit-learn: For metrics and utilities

These dependencies are listed in the project's `requirements.txt` file. 