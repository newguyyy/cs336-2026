from cs336_basics import nn_utils
from cs336_basics.tokenizer import Tokenizer
import argparse
from pathlib import Path
import torch
def parse_args():
    parser = argparse.ArgumentParser(description="模型参数路径")
    
    parser.add_argument("--model_path",type=str,help="checkpoint路径",required=True)
    parser.add_argument("--tokenizer_name",type = str,help="选择分词器的类型",required=True)
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()

    if args.tokenizer_name == "owt":
        data_vocab = "data/owt_vocab.json";data_merges = "data/owt_merges.txt";special_tokens = ["<|endoftext|>"]
    else:
        data_vocab = "data/tinestories_vocab.json";data_merges = "data/tinestories_merges.txt";special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(data_vocab,data_merges,special_tokens)
    
    checkpoint_path = Path(args.model_path) / "checkpoint.pt"
    hyperparam_path = Path(args.model_path) / "experiment_log.md"
 
    device = torch.device("cuda")
    x = nn_utils.TransformLM(10000,256,4,512,16,1344,10000,True,device=device)
    checkpoint = torch.load(checkpoint_path)
    x.load_state_dict(checkpoint["model"])
    msg = nn_utils.decoding(tokenizer,x,256,"Once upon a time there was a little boy named Ben. Ben loved to explore the world around him.  He saw many amazing things, like beautiful vases that were on display in a store. ",300,1,0.5)
    print(msg)