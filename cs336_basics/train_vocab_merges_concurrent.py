from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.train_vocab_merges import pre_tokenize,_merge
from tests.common import gpt2_bytes_to_unicode
from multiprocessing import Pool
from collections import Counter
import sys
import json
import time,tracemalloc
from tqdm import tqdm
import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
delimiter = re.escape("<|endoftext|>")

def worker(args) ->dict[tuple[bytes,...],int]:     # 把 I/O 和 decode 留在子进程里做，父进程只负责切分和汇总
    """
    Returns:
        预分词的后的[word,freq]
    """
    file_path,start,end  = args
    with open(file_path,"rb") as f:            # find_chunk_boundaries返回的是字节偏移,所以只能以二进制形式打开
        f.seek(start)
        chunk_bytes = f.read(end-start)
        chunk_str = chunk_bytes.decode("utf-8",errors = "ignore")
        segments = re.split(delimiter,chunk_str)    # "rb"是以二进制形式打开,需要decode为str才可以split
        return pre_tokenize(segments)

if __name__ == "__main__":
    t0 = time.time()
    t1 = t0
    tracemalloc.start()
    file_path = sys.argv[1] 
    vocab_size = int(sys.argv[2])
    output_prefix = sys.argv[3]
    print(f"File: {file_path}")
    num_workers =  2
    num_chunks = 40
    with open(file_path,"rb") as f:
        boundaries = find_chunk_boundaries(f,num_chunks,b"<|endoftext|>")  # num_chunks=40 是请求，不是保证。 实际块数取决于分隔符是否足够多、分布是否够散。
    # 准备任务列表
    tasks = [(file_path,start,end) for start,end in 
                zip(boundaries[:-1],boundaries[1:])]
    global_freqs = Counter()
    with Pool(num_workers) as pool:
        for d in tqdm(pool.imap_unordered(worker, tasks),
                total=len(tasks),
                desc="Pre-tokenizing"):
                    global_freqs.update(d)
        
    # global_freqs = Counter()
    # i = 1
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     with open(file_path, "rb") as f:
    #         f.seek(start)
    #         chunk_bytes = f.read(end - start)
    #     chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
    #     segments = re.split(delimiter, chunk_str)
    #     word_freqs = pre_tokenize(segments)
    #     global_freqs.update(word_freqs)
    #     print(f"  chunk done, {len(word_freqs)} words")
    #     elp = time.time() - t1
    #     print(f"{i} time elapse : {elp}\n")
    #     i+=1
    #     t1 = time.time()
    #     # 释放
    #     del chunk_bytes, chunk_str, segments, word_freqs
    #     gc.collect()

    print(f"Total word types: {len(global_freqs)}")
    min_freq = 5
    filtered = {word: freq for word, freq in global_freqs.items() if freq >= min_freq}
    print(f"After filtering (freq>={min_freq}): {len(filtered)}/{len(global_freqs)} word types")
    vocab, merges = _merge(filtered, ["<|endoftext|>"], vocab_size)

    # 内存 时间测量
    _,peak = tracemalloc.get_traced_memory()
    print(f"Peak memory: {peak / 1024**3:.2f} GB")
    tracemalloc.stop()
    elapsed = time.time() - t0
    print(f"Traceing took{elapsed:.1f}s")

    # 找出最长token
    longest = max(vocab.values(),key = len)
    print(f"最长 token: {longest.decode('utf-8',errors='replace')}")
    print(f"字节长度：{len(longest)}")
    # 词典序列化
    vocab_serializable = {k:list(v) for k,v in vocab.items()}
    with open(f"{output_prefix}_vocab.json","w") as f:
        json.dump(vocab_serializable,f)
    # merges -> 文本文件  使用gpt_bytes_unicode  把bytes映射为可打印的Unicode字符
    with open(f"{output_prefix}_merges.txt","w") as f:
        byte_to_char = gpt2_bytes_to_unicode()
        for (tok1,tok2) in merges:
            s1 = "".join(byte_to_char[b] for b in tok1)
            s2 = "".join(byte_to_char[b] for b in tok2)
            f.write(f"{s1} {s2}\n")