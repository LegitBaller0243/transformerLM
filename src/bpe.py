from collections import Counter
from multiprocessing import Pool
import os
from typing import BinaryIO
import regex as re
from bpe_accelerator import train
import pickle
import time
NUM_PROCESSES = 5

GPT_SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def train_bpe(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]
    ): 
    
    # return vocab: dict[int, bytes] and merges: list[tuple[bytes, bytes]]
    ETX = b"<|endoftext|>"
    with open(input_path, 'rb') as file:
        boundaries = find_chunk_boundaries(file, NUM_PROCESSES * 5, ETX)
    print(f"Boundaries: {boundaries}")
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    global_counts = Counter()

    # get chunks = num processes using find chunk boundaries
    with Pool(NUM_PROCESSES) as p:
        # for each chunk, run a pretokenizer -> list[str] tokens
        for counter in p.imap(pretokenize, tasks):
            global_counts.update(counter)

    # run train on it
    vocab, merges = train(global_counts, vocab_size, special_tokens)
    return vocab, merges

def pretokenize(args):
    src, start, end, special_tokens = args
    with open(src, 'rb') as file:
        file.seek(start)
        data = file.read(end - start)

    text = data.decode('utf-8', errors='ignore')
    if special_tokens:
        spcl_regex = "|".join(re.escape(token) for token in special_tokens)
        tokens = re.split(spcl_regex, text)
    else:
        tokens = [text]
    counts = Counter()
    for seg in tokens:
        if not seg: 
            continue
        counts.update(GPT_SPLIT_PATTERN.findall(seg))
    return counts
def serialize(vocab, vocab_path, merges, merges_path):
    ## serializes those into a filepath
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path) as f:
        pickle.dump(merges, f)
    

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
