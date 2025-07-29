from collections.abc import Iterable
from typing import Iterator
import regex as re


class Tokenizer:
    vocab:dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    priority_merge: dict[tuple[bytes,bytes],int]
    special_tokens: list[str] | None
    encoder_vocab:dict[bytes,int]
    
    def __init__(
        self,
        vocab:dict[int, bytes],
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        for i,merge in enumerate(merges):
            self.priority_merge[merge] = i
        self.encoder_vocab = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None):
        pass
    
    
    def encode(self,text: str) -> list[int]:
        final_token_bytes:list[bytes] = []
        # 使用sp_token分词
        if self.special_tokens:
            pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            pattern_for_capture = f"({pattern})"
            chunks = re.split(pattern,text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if not chunk:
                continue
            if self.special_tokens and chunk in self.special_tokens:
                final_token_bytes.append(chunk.encode("utf-8"))
            else:
                pre_token_bytes :list[list[bytes]] = []
                # 使用gpt-2分词规则
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT,chunk)]
                for token in tokens:
                    token_bytes = [bytes([b]) for b in token]
                    pre_token_bytes.append(token_bytes)
                for token in pre_token_bytes:
                    final_token_bytes.extend(self.bpe_merge(token))
        
        encoded_ids:list[int] = [self.encoder_vocab[token] for token in final_token_bytes]
        return encoded_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        pass
    
    
    def bpe_merge(self,token:list[bytes]):
        while True:
            near_bytes_pairs = list(zip(token,token[1:]))
            if not near_bytes_pairs:
                break
            max_bytes_pair = min(
                near_bytes_pairs,
                key =lambda p :self.priority_merge.get(p,float("inf")))
            if self.priority_merge.get(max_bytes_pair) is None:
                break
            a,b = max_bytes_pair
            new_token = []
            i = 0
            while i < len(token):
                # 查找第一次出现的目标合并对
                if i < len(token) - 1 and token[i] == a and token[i+1] == b:
                    # 找到了，将合并后的新字节添加到 new_token
                    new_token.append(a + b)
                    # 跳过两个旧字节
                    i += 2
                else:
                    # 没有找到，或者不是目标对，直接将当前字节添加到 new_token
                    new_token.append(token[i])
                    i += 1
            
            # 5. 用新生成的 token 列表覆盖旧的，准备进行下一轮合并
            token = new_token
        
        return token
            