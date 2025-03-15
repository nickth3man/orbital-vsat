"""
Dataset preparation script for VSAT benchmarks.

This script helps with downloading and preparing standard datasets
for benchmarking VSAT components.
"""

import os
import argparse
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
import subprocess
import urllib.request
import json
import tempfile
import csv

logger = logging.getLogger(__name__)

# Common benchmark datasets
DATASETS = {
    "librispeech-test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "type": "transcription",
        "description": "LibriSpeech test-clean dataset for transcription benchmarking"
    },
    "ami-corpus-sample": {
        "url": "https://github.com/pyannote/pyannote-db-odessa-ami/archive/refs/heads/master.zip",
        "type": "diarization",
        "description": "AMI Meeting Corpus sample for diarization benchmarking"
    },
    "librimix-sample": {
        "url": "https://github.com/JorisCos/LibriMix/archive/refs/heads/master.zip",
        "type": "separation",
        "description": "LibriMix sample for source separation benchmarking"
    }
}

def download_file(url: str, destination: str) -> None:
    """Download a file from a URL.
    
    Args:
        url: URL to download from
        destination: Destination file path
    """
    logger.info(f"Downloading {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress
    def report_progress(count, block_size, total_size):
        percent = count * block_size * 100 // total_size
        if percent % 5 == 0:  # Report every 5%
            logger.info(f"Download progress: {percent}%")
    
    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    logger.info(f"Downloaded {url} to {destination}")

def extract_archive(archive_path: str, extract_dir: str) -> None:
    """Extract an archive file.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract based on file extension
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    logger.info(f"Extracted {archive_path} to {extract_dir}")

def prepare_librispeech(dataset_dir: str, output_dir: str) -> None:
    """Prepare LibriSpeech dataset for transcription benchmarking.
    
    Args:
        dataset_dir: Path to extracted LibriSpeech dataset
        output_dir: Output directory for prepared dataset
    """
    logger.info(f"Preparing LibriSpeech dataset from {dataset_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all transcripts (*.trans.txt files)
    transcript_files = list(Path(dataset_dir).glob("**/*.trans.txt"))
    logger.info(f"Found {len(transcript_files)} transcript files")
    
    # Process transcript files
    metadata = []
    
    for transcript_file in transcript_files:
        # Read transcript file
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse line (format: ID TRANSCRIPT)
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id, transcript = parts
                    
                    # Find corresponding audio file
                    flac_path = transcript_file.parent / f"{file_id}.flac"
                    if flac_path.exists():
                        # Copy audio file to output directory
                        output_flac = Path(output_dir) / flac_path.name
                        shutil.copy(flac_path, output_flac)
                        
                        # Create transcript file
                        output_txt = output_flac.with_suffix('.txt')
                        with open(output_txt, 'w', encoding='utf-8') as f_out:
                            f_out.write(transcript)
                        
                        # Add to metadata
                        metadata.append({
                            'audio_filename': output_flac.name,
                            'transcript': transcript
                        })
    
    # Write metadata CSV
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['audio_filename', 'transcript'])
        writer.writeheader()
        writer.writerows(metadata)
    
    logger.info(f"Prepared {len(metadata)} items from LibriSpeech to {output_dir}")

def prepare_ami_corpus(dataset_dir: str, output_dir: str) -> None:
    """Prepare AMI Corpus sample for diarization benchmarking.
    
    Args:
        dataset_dir: Path to extracted AMI Corpus
        output_dir: Output directory for prepared dataset
    """
    logger.info(f"Preparing AMI Corpus sample from {dataset_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find annotation files (*.rttm)
    rttm_files = list(Path(dataset_dir).glob("**/*.rttm"))
    logger.info(f"Found {len(rttm_files)} RTTM files")
    
    # Find corresponding audio files
    processed = 0
    for rttm_file in rttm_files:
        # Extract meeting ID from RTTM file name
        meeting_id = rttm_file.stem
        
        # Look for audio file with same ID
        wav_file = None
        for ext in ['.wav', '.WAV']:
            potential_wav = list(Path(dataset_dir).glob(f"**/*{meeting_id}*{ext}"))
            if potential_wav:
                wav_file = potential_wav[0]
                break
        
        if wav_file and wav_file.exists():
            # Copy files to output directory
            output_wav = Path(output_dir) / wav_file.name
            output_rttm = Path(output_dir) / rttm_file.name
            
            shutil.copy(wav_file, output_wav)
            shutil.copy(rttm_file, output_rttm)
            
            processed += 1
    
    logger.info(f"Prepared {processed} items from AMI Corpus to {output_dir}")

def prepare_librimix(dataset_dir: str, output_dir: str) -> None:
    """Prepare LibriMix sample for source separation benchmarking.
    
    Args:
        dataset_dir: Path to extracted LibriMix
        output_dir: Output directory for prepared dataset
    """
    logger.info(f"Preparing LibriMix sample from {dataset_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find mixture directories
    mixture_dirs = [d for d in Path(dataset_dir).glob("**/wav8k/min/test") if d.is_dir()]
    
    if not mixture_dirs:
        # Try to find the LibriMix generation scripts
        scripts_dir = list(Path(dataset_dir).glob("**/scripts"))[0] if list(Path(dataset_dir).glob("**/scripts")) else None
        
        if scripts_dir:
            logger.info(f"Found LibriMix scripts at {scripts_dir}, running minimal dataset generation")
            
            # Generate a minimal dataset using the LibriMix scripts
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process script to generate a small sample
                script_path = scripts_dir / "create_librimix.sh"
                
                # Create a modified script for a small test set
                modified_script = os.path.join(temp_dir, "create_minimal_librimix.sh")
                with open(script_path, 'r') as f_in, open(modified_script, 'w') as f_out:
                    for line in f_in:
                        # Modify script parameters for a small dataset
                        if "wget" in line and "LibriSpeech" in line:
                            f_out.write(line.replace("train-clean-100", "test-clean").replace("dev-clean", "test-clean"))
                        elif "n_src=" in line:
                            f_out.write("n_src=2\n")  # Only use 2 sources
                        elif "metadata_dir=" in line:
                            f_out.write(f"metadata_dir={temp_dir}/metadata\n")
                        elif "librispeech_dir=" in line:
                            f_out.write(f"librispeech_dir={temp_dir}/LibriSpeech\n")
                        elif "librimix_outdir=" in line:
                            f_out.write(f"librimix_outdir={temp_dir}/LibriMix\n")
                        else:
                            f_out.write(line)
                
                # Make script executable
                os.chmod(modified_script, 0o755)
                
                # Run script
                logger.info(f"Running LibriMix generation script: {modified_script}")
                subprocess.run([modified_script], check=True, shell=True)
                
                # Find the generated mixtures
                mixture_dir = Path(temp_dir) / "LibriMix/wav8k/min/test"
                if mixture_dir.exists():
                    # Process the generated mixtures
                    process_librimix_directory(mixture_dir, output_dir)
                else:
                    logger.error(f"Failed to generate LibriMix sample dataset at {mixture_dir}")
        else:
            logger.error(f"Could not find LibriMix mixture directories or generation scripts")
    else:
        # Process existing mixtures
        logger.info(f"Found LibriMix mixture directories: {mixture_dirs}")
        for mixture_dir in mixture_dirs:
            process_librimix_directory(mixture_dir, output_dir)

def process_librimix_directory(mixture_dir: Path, output_dir: str) -> None:
    """Process LibriMix mixture directory.
    
    Args:
        mixture_dir: Path to LibriMix mixture directory
        output_dir: Output directory for prepared dataset
    """
    # Find mix files
    mix_files = list(mixture_dir.glob("mix/**/*.wav"))
    logger.info(f"Found {len(mix_files)} mixture files")
    
    # Prepare metadata
    metadata = []
    processed = 0
    
    for mix_file in mix_files[:10]:  # Limit to 10 files for the sample
        # Extract mixture ID
        mix_id = mix_file.stem
        
        # Find corresponding source files
        source_files = []
        for source_dir in mixture_dir.glob("s*"):
            if source_dir.is_dir() and source_dir.name.startswith('s'):
                source_file = source_dir / f"{mix_id}.wav"
                if source_file.exists():
                    source_files.append(source_file)
        
        if source_files:
            # Copy files to output directory
            output_mix = Path(output_dir) / f"mixed_{mix_id}.wav"
            shutil.copy(mix_file, output_mix)
            
            output_sources = []
            for i, source_file in enumerate(source_files):
                output_source = Path(output_dir) / f"source_{mix_id}_{i+1}.wav"
                shutil.copy(source_file, output_source)
                output_sources.append(output_source.name)
            
            # Add to metadata
            metadata.append({
                'mixed_filename': output_mix.name,
                'source_filenames': ';'.join(output_sources)
            })
            processed += 1
    
    # Write metadata CSV
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mixed_filename', 'source_filenames'])
        writer.writeheader()
        writer.writerows(metadata)
    
    logger.info(f"Prepared {processed} items from LibriMix to {output_dir}")

def prepare_dataset(dataset_name: str, output_dir: str, download_dir: str = None) -> None:
    """Prepare a benchmark dataset.
    
    Args:
        dataset_name: Name of dataset to prepare
        output_dir: Output directory for prepared dataset
        download_dir: Directory to download dataset to (default: temporary directory)
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_info = DATASETS[dataset_name]
    logger.info(f"Preparing dataset: {dataset_name} ({dataset_info['description']})")
    
    # Create temporary directory if download_dir not provided
    if download_dir is None:
        download_dir = tempfile.gettempdir()
    
    os.makedirs(download_dir, exist_ok=True)
    
    # Download archive
    archive_path = os.path.join(download_dir, os.path.basename(dataset_info['url']))
    if not os.path.exists(archive_path):
        download_file(dataset_info['url'], archive_path)
    
    # Extract archive
    extract_dir = os.path.join(download_dir, f"{dataset_name}_extracted")
    if not os.path.exists(extract_dir):
        extract_archive(archive_path, extract_dir)
    
    # Prepare dataset based on type
    if dataset_info['type'] == 'transcription':
        prepare_librispeech(extract_dir, output_dir)
    elif dataset_info['type'] == 'diarization':
        prepare_ami_corpus(extract_dir, output_dir)
    elif dataset_info['type'] == 'separation':
        prepare_librimix(extract_dir, output_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_info['type']}")
    
    logger.info(f"Prepared dataset: {dataset_name} to {output_dir}")

def list_datasets() -> None:
    """List available datasets."""
    print("\nAvailable datasets:")
    print("-" * 80)
    for name, info in DATASETS.items():
        print(f"{name}:")
        print(f"  Type: {info['type']}")
        print(f"  Description: {info['description']}")
        print(f"  URL: {info['url']}")
        print()

def main():
    """Command-line entry point for dataset preparation."""
    parser = argparse.ArgumentParser(description="VSAT Dataset Preparation Tool")
    
    parser.add_argument('--dataset', type=str,
                       help="Name of dataset to prepare")
    
    parser.add_argument('--output', type=str, default="datasets",
                       help="Output directory for prepared dataset")
    
    parser.add_argument('--download-dir', type=str,
                       help="Directory to download dataset to")
    
    parser.add_argument('--list', action='store_true',
                       help="List available datasets")
    
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                       format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    if args.list:
        list_datasets()
        return
    
    if args.dataset:
        # Prepare dataset
        output_dir = os.path.join(args.output, args.dataset)
        prepare_dataset(args.dataset, output_dir, args.download_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 