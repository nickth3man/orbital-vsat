"""
Visualization tools for VSAT benchmark results.

This script provides functionality to visualize benchmark results.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_benchmark_results(results_file):
    """Load benchmark results from a file.
    
    Args:
        results_file: Path to benchmark results JSON file
        
    Returns:
        Dictionary containing benchmark results
    """
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_wer_results(results, output_dir):
    """Plot Word Error Rate benchmark results.
    
    Args:
        results: Dictionary containing WER benchmark results
        output_dir: Directory to save plot files
    """
    # Create DataFrame from results
    data = []
    for item in results['results']:
        data.append({
            'audio_file': os.path.basename(item['audio_file']),
            'wer': item['wer'],
            'word_count': len(item['reference'].split())
        })
    
    df = pd.DataFrame(data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot WER distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['wer'], kde=True)
    plt.title(f'WER Distribution (Avg: {results["average_wer"]:.2f}%)')
    plt.xlabel('Word Error Rate (%)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'wer_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot top N worst files
    top_n = min(10, len(df))
    worst_files = df.sort_values('wer', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='wer', y='audio_file', data=worst_files)
    plt.title(f'Top {top_n} Worst WER Files')
    plt.xlabel('Word Error Rate (%)')
    plt.ylabel('Audio File')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wer_worst_files.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot WER vs file duration (if available in results)
    if 'duration' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='duration', y='wer', data=df)
        plt.title('WER vs Audio Duration')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Word Error Rate (%)')
        plt.savefig(os.path.join(output_dir, 'wer_vs_duration.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Plot WER vs word count
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='word_count', y='wer', data=df)
    plt.title('WER vs Reference Word Count')
    plt.xlabel('Word Count')
    plt.ylabel('Word Error Rate (%)')
    plt.savefig(os.path.join(output_dir, 'wer_vs_word_count.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Generate a summary table as CSV
    df.sort_values('wer', ascending=False).to_csv(
        os.path.join(output_dir, 'wer_summary.csv'), index=False
    )
    
    print(f"WER visualization complete. Results saved to {output_dir}")

def plot_der_results(results, output_dir):
    """Plot Diarization Error Rate benchmark results.
    
    Args:
        results: Dictionary containing DER benchmark results
        output_dir: Directory to save plot files
    """
    # Create DataFrame from results
    data = []
    for item in results['results']:
        data.append({
            'audio_file': os.path.basename(item['audio_file']),
            'der': item['der'],
            'miss': item.get('miss', 0),
            'false_alarm': item.get('false_alarm', 0),
            'speaker_error': item.get('speaker_error', 0),
            'duration': item.get('duration', 0)
        })
    
    df = pd.DataFrame(data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot DER distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['der'], kde=True)
    plt.title(f'DER Distribution (Avg: {results["average_der"]:.2f}%)')
    plt.xlabel('Diarization Error Rate (%)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'der_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot top N worst files
    top_n = min(10, len(df))
    worst_files = df.sort_values('der', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='der', y='audio_file', data=worst_files)
    plt.title(f'Top {top_n} Worst DER Files')
    plt.xlabel('Diarization Error Rate (%)')
    plt.ylabel('Audio File')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'der_worst_files.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot error components if available
    if all(c in df.columns for c in ['miss', 'false_alarm', 'speaker_error']):
        # Prepare data for stacked bar chart
        error_components = df[['audio_file', 'miss', 'false_alarm', 'speaker_error']].copy()
        error_components = error_components.sort_values('miss', ascending=False).head(top_n)
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 8))
        error_components.set_index('audio_file')[['miss', 'false_alarm', 'speaker_error']].plot(
            kind='bar', stacked=True, figsize=(12, 8)
        )
        plt.title('DER Error Components')
        plt.xlabel('Audio File')
        plt.ylabel('Error Rate (%)')
        plt.legend(title='Error Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'der_error_components.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a pie chart of average error components
        plt.figure(figsize=(10, 10))
        avg_errors = [df['miss'].mean(), df['false_alarm'].mean(), df['speaker_error'].mean()]
        plt.pie(avg_errors, labels=['Miss', 'False Alarm', 'Speaker Error'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Average DER Error Components')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, 'der_error_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Generate a summary table as CSV
    df.sort_values('der', ascending=False).to_csv(
        os.path.join(output_dir, 'der_summary.csv'), index=False
    )
    
    print(f"DER visualization complete. Results saved to {output_dir}")

def plot_sdr_results(results, output_dir):
    """Plot Signal-to-Distortion Ratio benchmark results.
    
    Args:
        results: Dictionary containing SDR benchmark results
        output_dir: Directory to save plot files
    """
    # Create DataFrame from results
    data = []
    for item in results['results']:
        # Each item may have multiple sources
        base_filename = os.path.basename(item['mixed_file'])
        
        if isinstance(item['sdr'], list):
            # Multiple sources case
            for i, sdr in enumerate(item['sdr']):
                data.append({
                    'mixed_file': base_filename,
                    'source_index': i + 1,
                    'sdr': sdr,
                    'duration': item.get('duration', 0)
                })
        else:
            # Single SDR case (average)
            data.append({
                'mixed_file': base_filename,
                'source_index': 0,  # Use 0 to indicate average
                'sdr': item['sdr'],
                'duration': item.get('duration', 0)
            })
    
    df = pd.DataFrame(data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot SDR distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sdr'], kde=True)
    plt.title(f'SDR Distribution (Avg: {results["average_sdr"]:.2f} dB)')
    plt.xlabel('Signal-to-Distortion Ratio (dB)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'sdr_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot top N worst files
    df_file_avg = df.groupby('mixed_file')['sdr'].mean().reset_index()
    top_n = min(10, len(df_file_avg))
    worst_files = df_file_avg.sort_values('sdr', ascending=True).head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='sdr', y='mixed_file', data=worst_files)
    plt.title(f'Top {top_n} Worst SDR Files')
    plt.xlabel('Signal-to-Distortion Ratio (dB)')
    plt.ylabel('Mixed Audio File')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sdr_worst_files.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot SDR by source index (if multiple sources)
    if len(df['source_index'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='source_index', y='sdr', data=df[df['source_index'] > 0])
        plt.title('SDR by Source Index')
        plt.xlabel('Source Index')
        plt.ylabel('Signal-to-Distortion Ratio (dB)')
        plt.savefig(os.path.join(output_dir, 'sdr_by_source.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Plot SDR vs file duration (if available)
    if 'duration' in df.columns and df['duration'].sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='duration', y='sdr', data=df)
        plt.title('SDR vs Audio Duration')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Signal-to-Distortion Ratio (dB)')
        plt.savefig(os.path.join(output_dir, 'sdr_vs_duration.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Generate a summary table as CSV
    df.sort_values('sdr', ascending=True).to_csv(
        os.path.join(output_dir, 'sdr_summary.csv'), index=False
    )
    
    print(f"SDR visualization complete. Results saved to {output_dir}")

def visualize_results(results_file, output_dir=None):
    """Visualize benchmark results.
    
    Args:
        results_file: Path to benchmark results JSON file
        output_dir: Directory to save plot files (default: same directory as results_file)
    """
    # Load results
    results = load_benchmark_results(results_file)
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_file), 'visualizations')
    
    # Determine type of benchmark from results
    if 'average_wer' in results:
        plot_wer_results(results, output_dir)
    elif 'average_der' in results:
        plot_der_results(results, output_dir)
    elif 'average_sdr' in results:
        plot_sdr_results(results, output_dir)
    else:
        raise ValueError(f"Unknown benchmark type in results file: {results_file}")

def main():
    """Command-line entry point for results visualization."""
    parser = argparse.ArgumentParser(description="VSAT Benchmark Results Visualizer")
    
    parser.add_argument('--results', type=str, required=True,
                        help="Path to benchmark results JSON file")
    
    parser.add_argument('--output', type=str,
                        help="Directory to save visualization files")
    
    args = parser.parse_args()
    
    # Visualize results
    visualize_results(args.results, args.output)

if __name__ == "__main__":
    main() 