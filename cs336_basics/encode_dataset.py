import sys,numpy as np
import os
from tqdm import tqdm
from cs336_basics.tokenizer import Tokenizer

def encode_file(tokenizer,input_path,output_path):
    file_size = os.path.getsize(input_path)
    with open(output_path,"wb") as out_f, \
         open(input_path,'r') as in_f, \
         tqdm(total=file_size,desc = f"Encoding {os.path.basename(input_path)}",unit="B",unit_scale=True) as pbar:
        for line in in_f:
            ids = tokenizer.encode(line)
            np.array(ids,dtype=np.uint16).tofile(out_f)
            pbar.update(len(line.encode()))

    
if __name__ == "__main__":
    data_name = sys.argv[1]      # "owt" / "ts"
    if data_name == "owt":
        data_vocab = "data/owt_vocab.json";data_merges = "data/owt_merges.txt";special_tokens = ["<|endoftext|>"]
    else:
        data_vocab = "data/tinestories_vocab.json";data_merges = "data/tinestories_merges.txt";special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(data_vocab,data_merges,special_tokens)
    input_path = sys.argv[2] 
    output_path = sys.argv[3]
    encode_file(tokenizer,input_path,output_path) 