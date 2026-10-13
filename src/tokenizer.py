import regex as re
import pickle
from collections.abc import Iterable, Iterator
from bpe_accelerator import encode_
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens: list[str]=None):
        self.vocab = vocab
        self.encode_dict = {token: idx for _, (idx, token) in enumerate(self.vocab.items())} # bytes -> int
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None # list[str]
        self.GPT_SPLIT_PATTERN = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens: list[str]):
        ## retrieve serialized vocab and merges via pickle
        with (vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with (merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str):
        # Implementation for encoding text into tokens
        # get text
        if self.special_tokens:
            spcl_regex = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            tokens = re.split(spcl_regex, text)
        else:
            tokens = [text]

        # get rid of null seg
        chunks = [token for token in tokens if token]
        # call Rust function here
        spcl_tokens = set(self.special_tokens) if self.special_tokens else set()
        
        final_ids = encode_(chunks, self.encode_dict, self.merges, spcl_tokens)
        return final_ids
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Implementation for encoding an iterable of strings into tokens
        ## have to write in Python i think, follow the lgoic of encode
        ## yield token ids per string in iterable
        for text in iterable:
            text_ids = self.encode(text)
            yield from text_ids
    def decode(self, ids: list[int]):
        # Implementation for decoding tokens into text
        res = []
        # i think its just for id in ids: self.vocab 
        for idx in ids:
            res.append(self.vocab[idx])
        text = b"".join(res).decode('utf-8', errors='replace')
        return text
    
