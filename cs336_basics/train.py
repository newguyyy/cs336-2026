from cs336_basics import nn_utils

import torch
import argparse
import numpy as np
from numpy import memmap
from einops import rearrange
from jaxtyping import Bool, Float,Int
from torch import Tensor

import time,csv
from pathlib import Path
import math
# paragrams
def parse_args():
    parser = argparse.ArgumentParser(description="训练脚本")
    # data path
    parser.add_argument("--train_path",type=str,help="训练数据路径")
    parser.add_argument("--val_path",type=str,help="验证数据")
    # model paragramz
    parser.add_argument("--vocab_size",type=int,default=10000)
    parser.add_argument("--num_layers",type=int,default=48,help="层数")
    parser.add_argument("--context_length",type=int,default=1024)
    parser.add_argument("--d_model",type=int,default=1600)
    parser.add_argument("--num_heads",type=int,default=25)
    parser.add_argument("--d_ff",type=int,default=4288)
    parser.add_argument("--rope_theta",type=float,default=50000)
    parser.add_argument("--no_rope",action="store_true")
    
    # train params
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--total_steps",type=int,default=5000)
    parser.add_argument("--device",type=str,default="cpu")

    
    # optimizer hyperparam
    parser.add_argument("--lr_max",type=float,default=3e-4)
    parser.add_argument("--lr_min",type=float,default=3e-5)
    parser.add_argument("--weight_decay",type=float,default=0.1)
    parser.add_argument("--beta1",type=float,default=0.9)
    parser.add_argument("--beta2",type=float,default=0.95)
    parser.add_argument("--eps",type=float,default=1e-8)
    parser.add_argument("--max_l2_norm",type=float,default=1.0)
    parser.add_argument("--warmup_iters",type=int,default=100)
    parser.add_argument("--cosine_cycle_iters",type=int,default=None)

    # log params 
    parser.add_argument("--log_every",type=int,default=10)
    parser.add_argument("--val_every",type=int,default=100)
    parser.add_argument("--save_every",type=int,default=1000)
    parser.add_argument("--log_dir",type=str,default="test_log")

    parser.add_argument("--overfit",action="store_true",help="过拟合单个minibatch的调试模式")
        
    args = parser.parse_args()
    return args
def compute_loss(model,x: Int[Tensor," batch_size seq"],y: Int[Tensor, " batch_size seq"]):
        logits = model(x)
        logits = rearrange(logits," batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size")        
        y = rearrange(y," batch_size seq_len -> (batch_size seq_len)")
        return nn_utils.CrossEntropy(logits,y) 
    
if  __name__=="__main__":
    args = parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.total_steps 
    # load data
    train_data = np.memmap(args.train_path,dtype=np.uint16,mode="r")
    val_data   = np.memmap(args.val_path,dtype=np.uint16,mode="r")

    new_dir = Path("data/log") / args.log_dir
    new_dir.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    train_path = new_dir / "train_log.csv"
    val_path = new_dir / "val_log.csv" 
    with open(train_path,'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step","wall_time","train_loss"])
    with open(val_path,'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step","wall_time","val_loss"])
    
            
    # load the model and optimizer
    model = nn_utils.TransformLM(vocab_size=args.vocab_size,context_length=args.context_length,num_layers=args.num_layers,
                                 d_model=args.d_model,num_heads=args.num_heads,d_ff=args.d_ff,rope_theta=args.rope_theta,device=args.device,rope=not args.no_rope)
    optimizer = nn_utils.AdamW(model.parameters(),lr=0.0,weight_decay=args.weight_decay,betas=(args.beta1,args.beta2),eps=args.eps)

    if args.overfit:
        x,y = nn_utils.DataLoading(train_data,args.batch_size,args.context_length,device=args.device)
        
    # train loop
    for t in range(1,args.total_steps + 1):

        lr = nn_utils.lr_cosine_schedule(t,args.lr_max,args.lr_min,args.warmup_iters,args.cosine_cycle_iters)
        optimizer.param_groups[0]["lr"] = lr
        
        # get a sampled batch
        if not args.overfit:
            x,y = nn_utils.DataLoading(train_data,args.batch_size,args.context_length,device=args.device)

        # forward and loss
        loss  = compute_loss(model,x,y)            # type: ignore

        # backward and update
        optimizer.zero_grad()
        loss.backward()
        nn_utils.gradient_clipping(model.parameters(),args.max_l2_norm)
        optimizer.step()
        
        if t % args.log_every == 0:
            elapsed = time.perf_counter() - start
            with open(train_path,'a',newline="") as f:
                writer = csv.writer(f)
                writer.writerow([t,elapsed,loss.item()])

        if t % args.val_every == 0:
            with torch.no_grad():
                val_x,val_y = nn_utils.DataLoading(val_data,args.batch_size,args.context_length,device=args.device)
                val_loss = compute_loss(model,val_x,val_y)
            elapsed = time.perf_counter() - start
            with open(val_path,'a',newline="") as f:
                writer = csv.writer(f)
                writer.writerow([t,elapsed,val_loss.item()])

        if t % args.save_every == 0:
            nn_utils.save_checkpoint(model,optimizer,t,(new_dir / "checkpoint.pt"))
            print(f"checkpoint saved at step {t}")

    with torch.no_grad():
        last_val_x,last_val_y = nn_utils.DataLoading(val_data,args.batch_size,args.context_length,device=args.device)
        last_val_loss = compute_loss(model,last_val_x,last_val_y).item()

    nolog_list = ["train_path","val_path","log_every","val_every","save_every","log_dir","overfit"]
    param_dict = { k:v for k,v in vars(args).items() if k not in nolog_list}
    param_dict["val_loss"] = last_val_loss
    param_dict["ppl"] = math.exp(last_val_loss) 
    param_dict["elapsed"] = time.perf_counter() - start
    param_dict = { k:str(v) for k,v in param_dict.items()}                 # 遍历dict默认得到键，遍历dict.items()才得到键值

    experiment_log_path = new_dir / "experiment_log.md"
    if not experiment_log_path.exists():
        with open(experiment_log_path,'w') as f:
            f.write("|"+"|".join(param_dict.keys())+"|\n")
            f.write("|"+"|".join(["---"]*len(param_dict))+"|\n")
    with open(experiment_log_path,'a') as f:
        f.write("|"+"|".join(param_dict.values())+"|\n")
