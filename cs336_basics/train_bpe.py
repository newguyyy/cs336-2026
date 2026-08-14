import regex as re
from tqdm import tqdm
import os


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pre_tokenize(segments:list[str]) -> dict[tuple[bytes,...],int]:
    pre_token = {}          #dict[tuple[bytes,...],int]
    for segment in segments:
        if not segment:           #跳过空字符
            continue
        for token in re.finditer(PAT,segment):
            token_encode  = token.group().encode()             #finditer返回是一个match对象,.group才能去出内容
            token_bytes = tuple(bytes([b]) for b in token_encode)
            pre_token[token_bytes] = pre_token.get(token_bytes,0) + 1
    return pre_token
def _merge(pre_token: dict[tuple[bytes,...],int],
           special_tokens: list[str],
           vocab_size: int, 
    )->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    print(f"[merge] 开始合并: {len(pre_token)} word types → target vocab {vocab_size}")
    print(f"[merge] 预计合并 {vocab_size - 256 - len(special_tokens)} 轮")

    vocab = {}           # dict[int, bytes]
    for i in range(256):
        vocab[i] = bytes([i])
    for spe_tok in special_tokens:
        vocab[len(vocab)] = bytes(spe_tok,"utf-8")
    merges = []
    pair_counts = {}       # dict[tuple[bytes,bytes],int]    总频率
    pair_to_words = {}           # dict[(bytes,bytes),set[tuple[bytes,...]]]   哪些单词包含它
    for word,freq in pre_token.items():
        for i in range(len(word)-1):
            pair = (word[i],word[i+1])
            pair_counts[pair] = pair_counts.get(pair,0) + freq
            pair_to_words.setdefault(pair,set()).add(word)
    try:
        total_round = vocab_size - len(vocab)
        with tqdm(total=total_round,desc = "Merging") as pbar:
            #   选取出现频率最高的pair进行merge
            while(len(vocab)<vocab_size):
                if not pair_counts:
                    break
                best_pair = max(pair_counts,key = lambda p: (pair_counts[p],p))       # max遍历字典返回的是key
                merged_pair = best_pair[0] + best_pair[1]
                merges.append(best_pair)
                vocab[len(vocab)] = merged_pair
                for word in list(pair_to_words[best_pair]):
                    f = pre_token[word]
                    pairs = []
                    # 搜集所有pair
                    for i in range(len(word)-1):
                        pairs.append((word[i],word[i+1]))
                    # 减掉旧word的所有贡献
                    for pair in pairs:
                        pair_counts[pair] -= f
                        pair_to_words[pair].discard(word)
                        if pair_counts[pair]==0:
                            del pair_counts[pair]
                            del pair_to_words[pair]
                    
                    # 构建new word
                    new_word = []             
                    j = 0
                    while j < len(word):
                        if j != len(word) - 1 and (word[j],word[j+1]) == best_pair:
                            new_word.append(merged_pair)
                            j+=2
                            continue
                        new_word.append(word[j])
                        j+=1

                    del pre_token[word]
                    pre_token[tuple(new_word)] = pre_token.get(tuple(new_word), 0) + f
                    # 增加新的word的pair
                    for i in range(len(new_word)-1):
                        pair = (new_word[i],new_word[i+1])
                        pair_counts[pair] = pair_counts.get(pair,0) + f
                        pair_to_words.setdefault(pair,set()).add(tuple(new_word))
                pbar.update(1)
    except Exception as e:
        print(f"[merge] 崩溃于 vocab={len(vocab)}, pairs={len(pair_counts)}")
        # 如果 best_pair 有值也打印出来
        import traceback
        traceback.print_exc()
        raise
    return vocab,merges
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path,"r") as f:
        text = f.read()
        # 预分词
        delimiter = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(delimiter,text)
        pre_token = pre_tokenize(segments)          #dict[tuple[bytes,...],int]
        # 词典初始化
        return _merge(pre_token,special_tokens,vocab_size)                                                                                    