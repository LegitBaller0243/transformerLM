"""Run from the repository root with python -m src.experiments.tokenizer.exp_tokenizer."""
import argparse
import cProfile
import json
import pickle
import pstats
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
from src.bpe import train_bpe
from src.tokenizer import Tokenizer
DATASETS = {
    'tinystories': ('TinyStoriesV2-GPT4-train.txt', 'TinyStoriesV2-GPT4-valid.txt'),
    'openwebtext': ('owt_train.txt', 'owt_valid.txt'),
}


def train_job(train_path, vocab_size, special_tokens, output, profile):
    profiler = cProfile.Profile() if profile else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        vocab, merges = train_bpe(str(train_path), vocab_size, special_tokens)
    finally:
        if profiler:
            profiler.disable()
    seconds = time.perf_counter() - start
    if profiler:
        profiler.dump_stats(str(output / 'train.prof'))
        pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(20)
    return vocab, merges, seconds


def save_tokenizer(output, vocab, merges):
    for name, value in [('vocab.pkl', vocab), ('merges.pkl', merges)]:
        with (output / name).open('wb') as destination:
            pickle.dump(value, destination)


def check_roundtrip(tokenizer, valid_path):
    with valid_path.open(encoding='utf-8') as source:
        text = source.read(10000)
    if not text:
        raise ValueError('Validation sample is empty')
    ids = tokenizer.encode(text)  # also warms up the encoder before timing
    if tokenizer.decode(ids) != text:
        raise AssertionError('Held-out sample failed encode/decode round trip')
    return text, ids


def benchmark_encoding(tokenizer, text, ids):
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        measured_ids = tokenizer.encode(text)
        durations.append(time.perf_counter() - start)
        assert measured_ids == ids
    seconds = statistics.median(durations)
    byte_count = len(text.encode('utf-8'))
    return {
        'sample_bytes': byte_count, 'sample_tokens': len(ids),
        'bytes_per_token': byte_count / len(ids), 'encode_seconds': durations,
        'median_encode_seconds': seconds, 'encode_bytes_per_second': byte_count / seconds,
        'encode_tokens_per_second': len(ids) / seconds, 'roundtrip_passed': True,
    }


def save_metrics(output, result):
    formatted = json.dumps(result, indent=2)
    (output / 'metrics.json').write_text(formatted + '\n')
    print(formatted, flush=True)


def run(args):
    train_name, valid_name = DATASETS[args.dataset]
    train_path, valid_path = args.data_dir / train_name, args.data_dir / valid_name
    for path in (train_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir / args.dataset / str(args.vocab_size)
    output.mkdir(parents=True, exist_ok=True)
    special_tokens = ['<|endoftext|>']

    print(f'Training {args.dataset}, vocabulary={args.vocab_size}', flush=True)
    vocab, merges, seconds = train_job(train_path, args.vocab_size, special_tokens, output, args.profile)

    save_tokenizer(output, vocab, merges)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    
    text, ids = check_roundtrip(tokenizer, valid_path)
    result = {
        'dataset': args.dataset, 'train_path': str(train_path), 'validation_path': str(valid_path),
        'requested_vocab_size': args.vocab_size, 'actual_vocab_size': len(vocab), 'merge_count': len(merges),
        'training_seconds': seconds, 'training_profiled': args.profile, 'python_version': sys.version,
        **benchmark_encoding(tokenizer, text, ids),
    }
    save_metrics(output, result)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/tokenizer')
    parser.add_argument('--profile', action='store_true', help='Profile training in the parent Python process')
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error('Vocabulary size must be at least 256')
    run(args)


if __name__ == '__main__':
    main()
