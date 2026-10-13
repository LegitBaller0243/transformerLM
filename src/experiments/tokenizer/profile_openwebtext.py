"""Diagnose training on a fixed 1 GB OpenWebText prefix, ending at a document boundary."""
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import psutil

from bpe_accelerator import train
from src.bpe import NUM_PROCESSES, find_chunk_boundaries, pretokenize

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'artifacts/openwebtext_sample_1gb'
SPECIAL = '<|endoftext|>'


def make_sample():
    with (ROOT / 'data/owt_train.txt').open('rb') as source:
        data = source.read(1_000_000_000)
    delimiter = SPECIAL.encode('utf-8')
    end = data.rfind(delimiter)
    if end < 0:
        raise ValueError('No endoftext delimiter found in the first 1 GB; inspect the data format')
    data = data[:end + len(delimiter)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'sample.txt'
    path.write_bytes(data)
    print(f'Sample: {len(data):,} bytes, {data.count(delimiter):,} document delimiters', flush=True)
    return path


def training_worker(path, connection):
    """Reuse the existing training stages without changing their implementation."""
    connection.send(('stage', 'chunk_boundaries'))
    with path.open('rb') as source:
        boundaries = find_chunk_boundaries(source, NUM_PROCESSES, SPECIAL.encode())
    connection.send(('boundaries', boundaries))
    tasks = [(str(path), start, end, [SPECIAL]) for start, end in zip(boundaries[:-1], boundaries[1:])]
    durations = {}
    try:
        connection.send(('stage', 'pretokenization'))
        start = time.perf_counter()
        with mp.Pool(NUM_PROCESSES) as pool:
            results = pool.starmap(pretokenize, tasks)
        durations['pretokenization'] = time.perf_counter() - start

        connection.send(('stage', 'aggregation'))
        start = time.perf_counter()
        counts = Counter()
        for result in results:
            counts.update(result)
        durations['aggregation'] = time.perf_counter() - start

        connection.send(('stage', 'rust_training'))
        start = time.perf_counter()
        vocab, merges = train(counts, 32000, [SPECIAL])
        durations['rust_training'] = time.perf_counter() - start
        connection.send(('result', {
            'stage_seconds': durations, 'unique_pretokens': len(counts),
            'total_pretoken_occurrences': sum(counts.values()),
            'actual_vocab_size': len(vocab), 'merge_count': len(merges),
        }))
    finally:
        connection.close()


def main():
    path = make_sample()
    receive, send = mp.Pipe(duplex=False)
    worker = mp.Process(target=training_worker, args=(path, send))
    worker.start()
    send.close()
    pipe_open = True
    stage = 'startup'
    peaks = {}
    result = None
    measurement_error = None
    process = psutil.Process(worker.pid)
    try:
        while worker.is_alive() or (pipe_open and receive.poll()):
            while pipe_open and receive.poll():
                try:
                    kind, value = receive.recv()
                except EOFError:
                    pipe_open = False
                    break
                if kind == 'stage':
                    stage = value
                    print(f'Stage: {stage}', flush=True)
                elif kind == 'boundaries':
                    print(f'Boundaries: {value}', flush=True)
                else:
                    result = value
            try:
                processes = [process, *process.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
                peaks[stage] = max(peaks.get(stage, 0), rss)
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, PermissionError):
                measurement_error = 'OS denied process-memory inspection'
            time.sleep(0.1)
    except KeyboardInterrupt:
        for child in process.children(recursive=True):
            child.terminate()
        worker.terminate()
        raise
    finally:
        worker.join()
        receive.close()
    if result is None or worker.exitcode != 0:
        raise RuntimeError(f'Training failed, exit code {worker.exitcode}')
    result['sample_bytes'] = path.stat().st_size
    result['sampled_peak_process_tree_rss_mb'] = {key: round(value / 1e6, 1) for key, value in peaks.items()}
    result['memory_measurement_note'] = measurement_error or '100 ms samples; sums RSS, so shared pages may be counted more than once'
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
