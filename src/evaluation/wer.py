"""
Word Error Rate (WER) calculation module.

This module provides functionality for calculating Word Error Rate,
a common metric for evaluating the quality of transcriptions.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
from Levenshtein import distance as levenshtein_distance

logger = logging.getLogger(__name__)

class WordErrorRate:
    """Class for calculating Word Error Rate (WER)."""
    
    def __init__(self, case_sensitive: bool = False, ignore_punctuation: bool = True):
        """Initialize the WER calculator.
        
        Args:
            case_sensitive: Whether to consider letter case when comparing words
            ignore_punctuation: Whether to ignore punctuation when comparing words
        """
        self.case_sensitive = case_sensitive
        self.ignore_punctuation = ignore_punctuation
        
        if self.ignore_punctuation:
            self.punctuation_pattern = re.compile(r'[^\w\s]')
        
        logger.debug(f"Initialized WER calculator (case_sensitive={case_sensitive}, "
                    f"ignore_punctuation={ignore_punctuation})")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for WER calculation.
        
        Args:
            text: Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase if not case-sensitive
        if not self.case_sensitive:
            text = text.lower()
        
        # Remove punctuation if requested
        if self.ignore_punctuation:
            text = self.punctuation_pattern.sub('', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def calculate(self, reference: str, hypothesis: str) -> Dict[str, Union[float, int]]:
        """Calculate Word Error Rate between reference and hypothesis texts.
        
        Args:
            reference: Reference (ground truth) text
            hypothesis: Hypothesis (transcribed) text
            
        Returns:
            Dict[str, Union[float, int]]: Dictionary with WER metrics including:
                - wer: Word Error Rate as a percentage
                - substitutions: Number of substituted words
                - deletions: Number of deleted words
                - insertions: Number of inserted words
                - total_words: Total number of words in reference
        """
        # Preprocess texts
        reference = self.preprocess_text(reference)
        hypothesis = self.preprocess_text(hypothesis)
        
        # Split into words
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        
        # Calculate Levenshtein distance
        operations = self._get_edit_operations(reference_words, hypothesis_words)
        
        # Count operation types
        substitutions = operations.count('substitution')
        deletions = operations.count('deletion')
        insertions = operations.count('insertion')
        
        # Calculate WER
        total_words = len(reference_words)
        if total_words == 0:
            # Handle empty reference case
            if len(hypothesis_words) == 0:
                wer = 0.0
            else:
                wer = 1.0
        else:
            wer = (substitutions + deletions + insertions) / total_words
        
        # Prepare results
        results = {
            'wer': wer * 100,  # as percentage
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'total_words': total_words
        }
        
        logger.debug(f"WER calculation: {results}")
        return results
    
    def _get_edit_operations(self, reference_words: List[str], 
                           hypothesis_words: List[str]) -> List[str]:
        """Get the edit operations (substitution, deletion, insertion) between word lists.
        
        Args:
            reference_words: List of words in the reference text
            hypothesis_words: List of words in the hypothesis text
            
        Returns:
            List[str]: List of operations that transform reference to hypothesis
        """
        # Initialize the dynamic programming matrix
        dp = [[0 for _ in range(len(hypothesis_words) + 1)] 
             for _ in range(len(reference_words) + 1)]
        operations = [['' for _ in range(len(hypothesis_words) + 1)] 
                    for _ in range(len(reference_words) + 1)]
        
        # Fill the first row and column
        for i in range(len(reference_words) + 1):
            dp[i][0] = i
            if i > 0:
                operations[i][0] = 'deletion'
        
        for j in range(len(hypothesis_words) + 1):
            dp[0][j] = j
            if j > 0:
                operations[0][j] = 'insertion'
        
        # Fill the rest of the matrix
        for i in range(1, len(reference_words) + 1):
            for j in range(1, len(hypothesis_words) + 1):
                if reference_words[i-1] == hypothesis_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    operations[i][j] = 'match'
                else:
                    # Determine the operation with minimum cost
                    substitution_cost = dp[i-1][j-1] + 1
                    deletion_cost = dp[i-1][j] + 1
                    insertion_cost = dp[i][j-1] + 1
                    
                    min_cost = min(substitution_cost, deletion_cost, insertion_cost)
                    dp[i][j] = min_cost
                    
                    if min_cost == substitution_cost:
                        operations[i][j] = 'substitution'
                    elif min_cost == deletion_cost:
                        operations[i][j] = 'deletion'
                    else:
                        operations[i][j] = 'insertion'
        
        # Backtrack to get the operations
        i, j = len(reference_words), len(hypothesis_words)
        edit_operations = []
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and operations[i][j] in ['match', 'substitution']:
                edit_operations.append(operations[i][j])
                i -= 1
                j -= 1
            elif i > 0 and operations[i][j] == 'deletion':
                edit_operations.append('deletion')
                i -= 1
            elif j > 0 and operations[i][j] == 'insertion':
                edit_operations.append('insertion')
                j -= 1
            else:
                # Shouldn't happen, but just in case
                break
        
        # Reverse to get the operations in the right order
        edit_operations.reverse()
        return edit_operations
    
    def benchmark(self, reference_texts: List[str], hypothesis_texts: List[str]) -> Dict[str, Any]:
        """Run a benchmark on multiple text pairs.
        
        Args:
            reference_texts: List of reference texts
            hypothesis_texts: List of hypothesis texts
            
        Returns:
            Dict[str, Any]: Benchmark results including average WER and per-sample metrics
        """
        if len(reference_texts) != len(hypothesis_texts):
            raise ValueError("Number of reference and hypothesis texts must match.")
        
        results = []
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_words = 0
        
        for i, (reference, hypothesis) in enumerate(zip(reference_texts, hypothesis_texts)):
            result = self.calculate(reference, hypothesis)
            results.append(result)
            
            total_substitutions += result['substitutions']
            total_deletions += result['deletions']
            total_insertions += result['insertions']
            total_words += result['total_words']
        
        # Calculate aggregate WER
        if total_words == 0:
            avg_wer = 0.0
        else:
            avg_wer = 100 * (total_substitutions + total_deletions + total_insertions) / total_words
        
        benchmark_results = {
            'average_wer': avg_wer,
            'total_substitutions': total_substitutions,
            'total_deletions': total_deletions,
            'total_insertions': total_insertions,
            'total_words': total_words,
            'sample_results': results
        }
        
        logger.info(f"WER benchmark completed with {len(results)} samples. Average WER: {avg_wer:.2f}%")
        return benchmark_results 