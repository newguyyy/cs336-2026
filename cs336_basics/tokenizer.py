from tests.common import gpt2_bytes_to_unicode
from collections.abc import Iterable
import json
import regex as re


class   Tokenizer:
    def __init__(self,
                 vocab: dict[int,bytes],
                 merges: list[tuple[bytes,bytes]],
                 special_tokens:list[str] | None=None,
):
        self.vocab = vocab
        self.merges = merges
        # 预建rank索引
        self.merge_rank = {pair:i for i,pair in enumerate(self.merges)}
        self.special_tokens = special_tokens or []
        existing = set(self.vocab.values())

        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in existing:
                self.vocab[len(self.vocab)] = tok_bytes
        # 建vocab反向索引
        self.byte_to_id  = {v: k for k,v in self.vocab.items()}

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens:list[str] | None = None
):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
        same format that your BPE training code output) and (optionally) a list of special tokens. 
        """
        # 读vocab
        with open(vocab_filepath) as f:              # 默认为'r'方式打开
            raw_vocab = json.load(f)
            vocab = {int(k): bytes(v) for k,v in raw_vocab.items()}
        
        # 读merges
        merges = []
        # mapping between a printable Unicode string and every possible byte(an interger from 0 - 255)   反向映射
        char_to_bytes = {char: bytes([byte]) for byte,char in gpt2_bytes_to_unicode().items()}
        with open(merges_filepath) as f:
            for line in f:
                tok1_str,tok2_str = line.rstrip().split(" ")
                # 还原为bytes
                tok1 = b"".join(char_to_bytes[c] for c in tok1_str)
                tok2 = b"".join(char_to_bytes[c] for c in tok2_str)
                merges.append((tok1,tok2))
        return cls(vocab,merges,special_tokens)


    def encode(self,text: str) -> list[int]:
        # 预分词
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens,key=len,reverse=True)
            special_patter = "|".join(re.escape(tok) for tok in sorted_tokens)
            special_patter = f"({special_patter})"         # 特殊token成为捕获组  此时特殊token在split后会保留
            segments = re.split(special_patter,text)
        else:
            segments = [text]      # -> list

        token_encoded = [] 
        # 特殊token应该作为独立token ID保留在输出中   之前使用split切块是为了并行化
        for segment in segments:
            if self.special_tokens and segment in self.special_tokens:
                token_encoded.append(self.byte_to_id[segment.encode("utf-8")])
            else:
                for m in re.finditer(PAT,segment):
                    token_str = m.group()
                    token_bytes_tuple = tuple(bytes([t]) for t in token_str.encode("utf-8"))
                    # 遍历merges直到不能merge为止  encode 的merge步骤是对每个pre-token独立做的
                    # enumerate merges  -> enumerate pre-token, index使得可以快速查找最小rank的pair
                    token = list(token_bytes_tuple)
                    while True:
                        best_rank = float('inf')
                        best_pos = -1
                        for i in range(len(token)-1):
                            pair = (token[i],token[i+1])
                            rank = self.merge_rank.get(pair,float('inf'))
                            if rank < best_rank:
                                best_rank = rank
                                best_pos = i
                        if best_rank == float('inf'):
                            break
                        else:
                            token[best_pos] = token[best_pos]+token[best_pos+1] 
                            del token[best_pos+1]
                    token_encoded.extend(self.byte_to_id[t] for t in token)
        return token_encoded
        


    def encode_iterable(self,iterable: Iterable[str])  -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    def decode(self,ids: list[int]) -> str:
        # 先拼所有bytes，再一次性decode
        token_bytes = b"".join(self.vocab[id] for id in ids)
        return token_bytes.decode("utf-8",errors="replace")