import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE (Byte Pair Encoding) tokenizer on the given input text file.

    Parameters
    ----------
    input_path : str
        Path to a UTF-8 encoded text file containing training data for the BPE tokenizer.
        Each line is considered part of the corpus.

    vocab_size : int
        The total size of the final vocabulary (must include initial byte-level tokens,
        all merged tokens produced during training, and the given special tokens).

    special_tokens : list[str]
        A list of user-defined special tokens (e.g., ["<|endoftext|>", "<pad>"]) to be 
        added to the vocabulary. These tokens do NOT participate in merge decisions.

    num_processes : int, optional (default=8)
        Number of parallel processes used during pre-tokenization. Each process handles
        a chunk of the input corpus split at special token boundaries. More processes
        generally mean faster pre-tokenization.

    Returns
    -------
    vocab : dict[int, bytes]
        A dictionary mapping token IDs (integers) to token values (in bytes). The token 
        IDs should be assigned sequentially starting from 0.

    merges : list[tuple[bytes, bytes]]
        A list of BPE merge operations, where each tuple represents two byte-level tokens 
        that were merged together. The list should be ordered by merge time (first merge first).
    """

    # 1. Vocabulary Initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")


    # 2. Pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    task_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)
    
    # 3. Compute BPE merges
    merges : list[tuple[bytes, bytes]] = []
    pre_tokens_bytes: list[list[bytes]] = [token for chunk in chunk_results for token in chunk]
    counts = defaultdict(int)
    pair_to_indices = defaultdict(set)
    for idx, token in enumerate(pre_tokens_bytes):
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            counts[pair] += 1
            pair_to_indices[pair].add(idx)

    idx = len(vocab)
    while idx < vocab_size:
        if not counts:
            break
            
        max_pair: tuple[bytes, bytes] = None
        max_cnt= -1
        for pair, cnt in counts.items():
            if cnt > max_cnt:
                max_pair = pair
                max_cnt = cnt
            elif cnt == max_cnt:
                if max_pair is None or pair > max_pair:
                    max_pair = pair

        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                old_pair = (token[i], token[i+1])
                pair_to_indices[old_pair].discard(j)
                counts[old_pair] -= 1
                if counts[old_pair] == 0:
                    counts.pop(old_pair)
                    pair_to_indices.pop(old_pair, None)

            merged = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == a and token[i+1]==b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            pre_tokens_bytes[j]=merged

            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += 1
                pair_to_indices[pair].add(j)

    return vocab, merges

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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
                true_position = initial_position + found_at
                chunk_boundaries[bi] = true_position
                break

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[bytes]]:
    input_path, start, end, special_tokens = args
    """
    Processes a chunk of the input file and returns byte pair frequency counts.

    Args:
        input_path (str): The path of input file.
        start (int): Start byte offset of the chunk.
        end (int): End byte offset of the chunk.
        special_tokens (list[str]): List of special tokens that should not be merged across.

    Returns:
        pre_token_bytes (list[list[bytes]]): list of tokens, where each token is a list of bytes
    """

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    # 1. Remove special tokens by splitting the chunk at those tokens
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    documents = re.split(pattern, chunk)

    # 2. Pre-tokenize and count byte pair frequencies
    pre_tokens_bytes: list[list[bytes]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, doc)]
        for token in tokens:
            token_bytes = [bytes([b]) for b in token]
            pre_tokens_bytes.append(token_bytes)

    return pre_tokens_bytes

