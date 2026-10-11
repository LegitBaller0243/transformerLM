from collections import Counter
from multiprocessing import Pool
from pretokenization_example import find_chunk_boundaries
import regex as re
from bpe_accelerator import train
NUM_PROCESSES = 5

GPT_SPLIT_PATTERN = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

def train_bpe(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]
    ): 
    
    # return vocab: dict[int, bytes] and merges: list[tuple[bytes, bytes]]
    ETX = b"\x03"
    boundaries = find_chunk_boundaries(input_path, NUM_PROCESSES, ETX)

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    # get chunks = num processes using find chunk boundaries
    with Pool(NUM_PROCESSES) as p:
        # for each chunk, run a pretokenizer -> list[str] tokens
        res = p.starmap(pretokenize, tasks)
    ## combine all the res's

    global_counts = Counter()
    for counter in res:
        global_counts.update(counter)
    # run train on it

    pre_tokens = list(global_counts.elements())

    vocab, merges = train(pre_tokens, vocab_size)

    return vocab, merges

def pretokenize(src, start, end, special_tokens):
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
    