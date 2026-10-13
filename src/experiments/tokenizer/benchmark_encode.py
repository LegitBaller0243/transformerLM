"""Benchmark saved OpenWebText tokenizer on 10 KB, 100 KB, 1 MB, and 10 MB prefixes."""
import json
import pickle
import statistics
import time
from pathlib import Path

from src.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / 'artifacts/tokenizer/openwebtext/32000'
SIZES = (10_000, 100_000, 1_000_000, 10_000_000)


def main():
    with (ARTIFACTS / 'vocab.pkl').open('rb') as source:
        vocab = pickle.load(source)
    with (ARTIFACTS / 'merges.pkl').open('rb') as source:
        merges = pickle.load(source)
    tokenizer = Tokenizer(vocab, merges, ['<|endoftext|>'])
    with (ROOT / 'data/owt_valid.txt').open('rb') as source:
        data = source.read(max(SIZES))
    if len(data) < max(SIZES):
        raise ValueError('Validation file is shorter than 10 MB')
    results = []
    output = ARTIFACTS / 'encoding_benchmark.json'
    for size in SIZES:
        # Ignore only a potentially incomplete final UTF-8 character at the byte cutoff.
        text = data[:size].decode('utf-8', errors='ignore')
        byte_count = len(text.encode('utf-8'))
        print(f'{size:,} bytes: warm-up', flush=True)
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text:
            raise AssertionError(f'Round trip failed at {size} bytes')
        durations = []
        for repeat in range(3):
            print(f'{size:,} bytes: timed run {repeat + 1}/3', flush=True)
            start = time.perf_counter()
            measured_ids = tokenizer.encode(text)
            durations.append(time.perf_counter() - start)
            assert measured_ids == ids
            del measured_ids
        median = statistics.median(durations)
        result = {
            'requested_bytes': size, 'actual_bytes': byte_count, 'tokens': len(ids),
            'seconds': durations, 'median_seconds': median,
            'bytes_per_second': byte_count / median, 'tokens_per_second': len(ids) / median,
        }
        results.append(result)
        output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        del ids


if __name__ == '__main__':
    main()
