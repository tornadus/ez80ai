#!/usr/bin/env python3
"""
Text encoding utilities for eZ80 character-level language models.

Provides trigram-based query encoding, context encoding for recent output
characters, training example generation, and data loading helpers.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


class TrigramEncoder:
    """Encode text into trigram hash buckets (integer-friendly, no normalization)."""

    def __init__(self, num_buckets: int = 128):
        self.num_buckets = num_buckets

    def _hash_trigram(self, trigram: str) -> int:
        """Hash a trigram to a bucket index."""
        h = 0
        for c in trigram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, text: str) -> np.ndarray:
        """Encode text into bucket counts (raw counts)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)
        text = text.lower()
        text = ' ' + text + ' '  # Pad for boundary trigrams

        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            bucket = self._hash_trigram(trigram)
            vec[bucket] += 1.0

        return vec


class ContextEncoder:
    """Encode recent output characters into hash buckets (integer-friendly)."""

    def __init__(self, num_buckets: int = 128, context_len: int = 8):
        self.num_buckets = num_buckets
        self.context_len = context_len

    def _hash_ngram(self, ngram: str, offset: int = 0) -> int:
        """Hash an n-gram with position offset."""
        h = offset * 7
        for c in ngram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, recent_chars: str) -> np.ndarray:
        """Encode recent output characters (raw counts)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)

        # Pad to context_len
        recent = recent_chars[-self.context_len:].lower()
        recent = recent.rjust(self.context_len)

        # Hash character n-grams with position info
        for n in [1, 2, 3]:  # Unigrams, bigrams, trigrams
            for i in range(len(recent) - n + 1):
                ngram = recent[i:i+n]
                bucket = self._hash_ngram(ngram, offset=i)
                vec[bucket] += 1.0

        return vec


def create_training_examples(query: str, response: str,
                            query_encoder: TrigramEncoder,
                            context_encoder: ContextEncoder,
                            char_to_idx: Callable[[str], int],
                            eos_idx: int) -> List[Tuple[np.ndarray, int]]:
    """
    Create training examples from a (query, response) pair.

    For response "hello", creates:
    - (query + context(""), 'h')
    - (query + context("h"), 'e')
    - (query + context("he"), 'l')
    - ...
    - (query + context("hello"), EOS)
    """
    examples = []
    query_vec = query_encoder.encode(query)

    # Add EOS to response
    response_with_eos = response + "\x00"

    output_so_far = ""
    for char in response_with_eos:
        # Encode current context
        context_vec = context_encoder.encode(output_so_far)

        # Combine query and context
        full_input = np.concatenate([query_vec, context_vec])

        # Target is next character (or EOS)
        target = char_to_idx(char) if char != "\x00" else eos_idx

        examples.append((full_input, target))
        output_so_far += char

    return examples


def parse_pair(line: str) -> Optional[Tuple[str, str]]:
    """Parse a single line into (query, response) or None if invalid."""
    line = line.strip()
    if '|' not in line:
        return None

    parts = line.split('|', 1)
    if len(parts) != 2:
        return None

    query = parts[0].strip().upper()
    response = parts[1].strip().upper()

    if len(query) >= 2 and len(response) >= 1:
        # Truncate smartly
        if len(query) > 60:
            query = query[:60].rsplit(' ', 1)[0] if ' ' in query[40:60] else query[:60]
        if len(response) > 50:
            response = response[:50].rsplit(' ', 1)[0] if ' ' in response[30:50] else response[:50]
        return (query, response)

    return None


def load_chunk(stdin, chunk_size: int = 0) -> List[Tuple[str, str]]:
    """Load up to chunk_size pairs from stdin (0 = all)."""
    pairs = []
    for line in stdin:
        pair = parse_pair(line)
        if pair:
            pairs.append(pair)
            if chunk_size > 0 and len(pairs) >= chunk_size:
                break
    return pairs
