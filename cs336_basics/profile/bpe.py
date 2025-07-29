from tests.adapters import run_train_bpe
import cProfile, pstats, io
from pstats import SortKey
import pathlib

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...
    FIXTURES_PATH = pathlib.Path("/Users/donkeykane/workspace/cs336/assignment1-basics/tests/fixtures")
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())