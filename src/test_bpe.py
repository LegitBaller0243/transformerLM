import os
import regex as re
from bpe_accelerator import train

def run_test():
    # 1. Create the Sennrich paper test corpus
    test_file = "test_corpus.txt"
    sample_text = (
        b"low low low low low\n"
        b"lower lower widest widest widest\n"
        b"newest newest newest newest newest newest\n"
    )
    
    with open(test_file, "wb") as f:
        f.write(sample_text)
        
    print(f"Created temporary corpus at '{test_file}'")

    # 2. Pretokenization step in Python (simulating what your multiprocessing pool outputs)
    # Read the file and pretokenize into a flat list of strings (Python list[str])
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple split or your regex pretokenization logic producing list of words/tokens
    pre_tokens = text.split() 
    
    vocab_size = 275  

    try:
        print(f"Passing {len(pre_tokens)} pre-tokens to Rust `train` function...")
        
        # 3. Pass the Python list of strings (which maps to Rust's Vec<String>)
        vocab, merges = train(pre_tokens, vocab_size)
        
        print("\n--- BPE Training Successful! ---")
        print(f"Total Vocab Size: {len(vocab)}")
        print(f"Total Merges Learned: {len(merges)}")
        
        print("\nLearned Merges:")
        for i, (p1, p2) in enumerate(merges):
            print(f"  {i+1}: {p1} + {p2}")
            
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up '{test_file}'.")

if __name__ == "__main__":
    run_test()
