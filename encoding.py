#!/usr/bin/env python3
"""
Text encoding utilities for eZ80 character-level language models.

Provides trigram-based query encoding, context encoding for recent output
characters, training example generation, and data loading helpers.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


class TrigramEncoder:
    """Encode query text into n-gram hash buckets (integer-friendly, no
    normalization).

    SPEC-DRIVEN (skew fix): the n-gram orders and hash params come from the model
    spec, so train.py and ez80research/evaluate.py reproduce intkernel.tokenize_query
    (and therefore the eZ80 build) EXACTLY. The old version hardcoded trigrams +
    mult=31/mask=0xFFFF, so changing query_ngram_orders/query_hash in the spec
    silently affected only the build/intkernel — a train-vs-build skew. Defaults
    reproduce the original trigram behavior byte-for-byte (pos_offset_mult=0, and
    h & (n-1) == h % n for power-of-two bucket counts)."""

    def __init__(self, num_buckets: int = 128, ngram_orders=(3,),
                 hash_mult: int = 31, hash_mask: int = 0xFFFF,
                 pos_offset_mult: int = 0, signed_hash: bool = False):
        self.num_buckets = num_buckets
        self.ngram_orders = tuple(ngram_orders)
        self.hash_mult = hash_mult
        self.hash_mask = hash_mask
        self.pos_offset_mult = pos_offset_mult
        self.signed_hash = signed_hash

    def _hash_ngram(self, ngram: str, pos: int) -> int:
        h = pos & self.hash_mask
        for c in ngram:
            h = (h * self.hash_mult + ord(c)) & self.hash_mask
        return h

    def encode(self, text: str) -> np.ndarray:
        """Encode text into bucket counts (raw counts, or signed count-sketch)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)
        text = ' ' + text.lower() + ' '  # 1-space pad both ends (boundary n-grams)
        for n in self.ngram_orders:
            for i in range(len(text) - n + 1):
                pos = (i * self.pos_offset_mult) & self.hash_mask
                h = self._hash_ngram(text[i:i + n], pos)
                # signed count-sketch: a high hash bit gives a ±1 sign so colliding
                # n-grams cancel in expectation instead of piling up bias.
                inc = -1.0 if (self.signed_hash and (h & 0x8000)) else 1.0
                vec[h & (self.num_buckets - 1)] += inc
        return vec


class ContextEncoder:
    """Encode recent output characters into n-gram hash buckets (integer-friendly).

    SPEC-DRIVEN (skew fix): n-gram orders / hash params come from the spec so
    train/eval reproduce intkernel.encode_context (and the eZ80 build) EXACTLY.
    Defaults reproduce the original [1,2,3] / mult=31 / offset*7 behavior byte-for-
    byte."""

    def __init__(self, num_buckets: int = 128, context_len: int = 8,
                 ngram_orders=(1, 2, 3), hash_mult: int = 31,
                 hash_mask: int = 0xFFFF, pos_offset_mult: int = 7,
                 signed_hash: bool = False):
        self.num_buckets = num_buckets
        self.context_len = context_len
        self.ngram_orders = tuple(ngram_orders)
        self.hash_mult = hash_mult
        self.hash_mask = hash_mask
        self.pos_offset_mult = pos_offset_mult
        self.signed_hash = signed_hash

    def _hash_ngram(self, ngram: str, pos: int) -> int:
        h = pos & self.hash_mask
        for c in ngram:
            h = (h * self.hash_mult + ord(c)) & self.hash_mask
        return h

    def encode(self, recent_chars: str) -> np.ndarray:
        """Encode recent output characters (raw counts, or signed count-sketch)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)
        recent = recent_chars[-self.context_len:].lower().rjust(self.context_len)
        for n in self.ngram_orders:
            for i in range(len(recent) - n + 1):
                pos = (i * self.pos_offset_mult) & self.hash_mask
                h = self._hash_ngram(recent[i:i + n], pos)
                inc = -1.0 if (self.signed_hash and (h & 0x8000)) else 1.0
                vec[h & (self.num_buckets - 1)] += inc
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
