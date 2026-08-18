import torch
import torch.nn as nn
from torch import Tensor,tensor
from torch import log,sum,exp,mean

import math
import numpy as np
import einops
from einops import einsum,reduce,rearrange
from jaxtyping import Bool, Float,Int
from collections.abc import Callable, Iterable

from typing import Optional
import numpy.typing as npt
from typing import IO, Any, BinaryIO
import os

from cs336_basics.tokenizer import Tokenizer

class ParameterWrapper(nn.Module):
    def __init__(self,*shape,device=None,dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape,device=device,dtype=dtype))

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int,   #final dimension of the input
                 out_features: int,    # final dimension of the output
                 device: torch.device | None = None,  #Device to store the parameters on
                 dtype: torch.dtype | None = None   # data type of parameters_
    ):
        super().__init__()
        
        self.out_features = out_features 
        # 创建 W 矩阵
        weight_data  = torch.empty(out_features,in_features,device = device,dtype = dtype)
        std = (2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(weight_data,mean=0.0,std = std,a = -3*std,b = 3*std)
        self.weight = nn.Parameter(weight_data)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        ans = x @ self.weight.T
        assert ans.shape[-1:] == (self.out_features,)
        return ans

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int, # Size of vocabulary
                 embedding_dim: int,    # Dimension of the embedding vector 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
        ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        #　创建 weight 矩阵
        weight_data = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        nn.init.trunc_normal_(weight_data,mean=0.0,std=1,a=-3,b=3)
        self.weight = nn.Parameter(weight_data)
    
    def forward(self,token_ids : torch.Tensor)   ->  torch.Tensor:
        ans = self.weight[token_ids]
        assert ans.shape == (token_ids.size(0),token_ids.size(1),self.embedding_dim)
        return ans

class RMSnorm(nn.Module):
    def __init__(self,
                 d_model: int ,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))     # 所有 token 位置共享的
        self.eps = eps
    def forward(self,x: Float[Tensor," ... d_model"]) -> Float[Tensor," ... d_model"]:
        # 防止平方溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # perform RMSNorm
        rms = x.pow(2).mean(dim = -1,keepdim=True).add(self.eps).sqrt()  
        result = x/rms*self.weight 
        assert result.shape == x.shape
        return result.to(in_dtype)
      
      
class SwiGLU(nn.Module):
    def __init__(self,d_model: int,d_ff: int,device=None,dtype=None):
        super().__init__()
        self.d_ff = d_ff
        std = (2/(d_model+d_ff))**0.5

        self.w1 = ParameterWrapper(d_ff,d_model,device=device,dtype=dtype)
        nn.init.trunc_normal_(self.w1.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
        self.w3 = ParameterWrapper(d_ff,d_model,device=device,dtype=dtype)
        nn.init.trunc_normal_(self.w3.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
        self.w2 = ParameterWrapper(d_model,d_ff,device=device,dtype=dtype)
        nn.init.trunc_normal_(self.w2.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def SiLU(self,x: torch.Tensor) -> torch.Tensor:
        return x*torch.sigmoid(x)
    def forward(self,x: Float[Tensor," ... d_model"])  ->  Float[Tensor," ... d_model"]:
        x_w1 = x@self.w1.weight.T 
        assert x_w1.shape[-2:] == (x.size(-2),self.d_ff)
        x_w3 = x@self.w3.weight.T 
        return self.SiLU(x_w1)*(x_w3)@self.w2.weight.T
            
        
class Rope(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor
    def __init__(self,theta: float,d_k: int,max_seq_len: int,device = None):
        super().__init__()
        indics = torch.arange(0,d_k,2)        # [0,2,4,....,d_k-2]  shape -> (d_k/2,)
        freqs = theta**(-indics/d_k)          # Θ^(-indics/d_k)    shape -> (d_k/2,)
        pos = torch.arange(max_seq_len)       # [0,1,2... max_seq_len]  shapes -> (max_seq_len)
        angles = einsum(pos,freqs,"i, j -> i j")
        cos = torch.cos(angles).to(device=device)
        sin = torch.sin(angles).to(device=device)

        self.register_buffer("cos",cos,persistent=False)
        self.register_buffer("sin",sin,persistent=False)
    def forward(self,x: Float[Tensor,"... seq_len d_k"],token_positions: Int[Tensor," ... seq_len"])  ->  Float[Tensor," ... seq_len d_k"]:
        x_pairs = rearrange(x,"... seq (d two) -> ... seq d two",two=2)

        x_even = x_pairs[...,0]     # (...,seq_len,d_k/2)
        x_odd  = x_pairs[...,1]     # (...,seq_len,d_k/2)

        x_even_new = x_even*self.cos[token_positions] - x_odd*self.sin[token_positions]     # x_even (...,seq_len,d_k/2)    cos[token_positions]  (...,seq_len,d_k/2)  *代表逐元素相乘
        x_odd_new = x_even*self.sin[token_positions] + x_odd*self.cos[token_positions]
        
        x_rotated = rearrange([x_even_new,x_odd_new],"two ... d -> ... (d two)")
        return x_rotated
    
def softmax(x:torch.Tensor,
            i: int, #apply softmax to the i-th dimension of the input tensor
            ):
    x_stab = x-x.max(i,keepdim=True).values
    return x_stab.exp()/x_stab.exp().sum(dim=i,keepdim=True)


def scaled_dot_product_attention(Q: Float[Tensor,"batch_size ... seq_len d_k"],
                                 K: Float[Tensor,"batch_size ... seq_len d_k"],
                                 V: Float[Tensor,"batch_size ... seq_len d_v"],
                                 mask: Bool[Tensor, "seq_len seq_len"] | None = None
                                 ) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_k = Q.size(-1)
    scores = einsum(Q,K,"... n d_k,... m d_k -> ... n m")/(d_k**0.5)
    assert scores.shape[-2:] == (Q.size(-2),K.size(-2))
    if mask is not None:
        scores = scores.masked_fill(mask==False,float("-inf"))
    attn = torch.softmax(scores,dim=-1)
    return einsum(attn,V,"... n m,... m d_v -> ... n d_v")
    


class MultiheadSelfAttention(nn.Module):
    def __init__(self,d_model: int,num_heads: int,max_seq_len: int | None= None, theta: float | None= None, rope=False,device = None,dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_control = rope
        if rope:
            self.max_seq_len = max_seq_len
            self.theta = theta
            self.rope = Rope(theta,d_model//num_heads,max_seq_len,device)  # type: ignore
        std = (2/(d_model+d_model))**0.5 
        for name in ["q_proj","k_proj", "v_proj", "output_proj"]:     
            wrapper = ParameterWrapper(d_model, d_model, device=device, dtype=dtype)
            nn.init.trunc_normal_(wrapper.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
            setattr(self, name, wrapper) 

    def forward(self,x: Float[Tensor,"... seq_len d_model"],token_positions: Float[Tensor," ... seq_len"] | None = None,
                ) -> Float[Tensor,"... seq_len d_model"]:
        Q = einsum(x,self.q_proj.weight," ... sql d_model, d d_model -> ... sql d")      #type: ignore
        Q = rearrange(Q," ... sql (h d_k) -> ... h sql d_k",h = self.num_heads)
        assert Q.shape[-3:] == (self.num_heads,Q.size(-2),self.d_model//self.num_heads)             # type: ignore
        K = einsum(x,self.k_proj.weight," ... sql d_model, d d_model -> ... sql d")      # type:ignore 
        K = rearrange(K," ... sql (h d_k) -> ... h sql d_k",h = self.num_heads)
        V = einsum(x,self.v_proj.weight," ... sql d_model, d d_model -> ... sql d")      # type: ignore
        V = rearrange(V," ... sql (h d_v) -> ... h sql d_v",h = self.num_heads)

        # rope
        if self.rope_control:
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
            
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len,seq_len,dtype = torch.bool,device=x.device))


        heads = scaled_dot_product_attention(Q,K,V,mask)     # shape (... num_heads seq_len d_v)   # type: ignore
        multiheads = rearrange(heads," ... h seq d_v -> ... seq (h d_v)")
        return einsum(multiheads,self.output_proj.weight,"... seq d,d_model d -> ... seq d_model")     # type: ignore


class TransformerBlock(nn.Module):
    def __init__(self,d_model: int,num_heads: int,d_ff: int, max_seq_len: int,theta: float,rope=False,device = None,dtype = None):  
        super().__init__()
        self.rope = rope
        self.ln1 = RMSnorm(d_model=d_model,device=device)          # for first sub-layer
        self.ln2 = RMSnorm(d_model=d_model,device=device)
        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff,device=device)
        self.attn = MultiheadSelfAttention(d_model=d_model,num_heads=num_heads,max_seq_len=max_seq_len,theta=theta,rope=rope,device=device)

    def forward(self,x: Float[Tensor," batch seq_len d_model"]) -> Float[Tensor,"batch seq_len d_model"]:
        seq_len = x.size(-2)
        if self.rope:
            token_position = torch.arange(seq_len,device=x.device)
            x = x + self.attn(self.ln1(x),token_position)
        else:
            x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))
        

class TransformLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 rope = True,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
    ):
        """
        vocab_size:  50,257
        context_length:  1,024
        num_layers:  48
        d_model:  1,600
        num_heads:  25    (d_k,d_v = d_model/num_heads = 64)
        d_ff:  4,288 (the nearest multiple of 64 to 8/3*1,600)

        transformer_accounting
        component                                 weight                                   matrix multiples                                                                                   FLOPs cost
                                                                                                                                                    (4e6+1.5e10+1e5+3e9+3e9+5e9+4e6+3e10+4e6+1.5e10)*48+4e6+1.6e11 = 3.5e12                
        token_embeddings                         (50257,1600) 
        
        transfomer_block*48:
            RMSnorm                              (1600,)                        (1024,1600)*(1600,) -> (1024,1600)  x/rms*self.weight                                                            4e6                      
            MultiheadSelfAttention:
                                    q_proj(1600,1600)  k_proj(1600,1600)                   (1024,1600)@(1600,1600) -> (1024,1600)*3  x@q_proj | x@k_proj | x@v_proj                              1.5e10
                                    
                                    v_proj(1600,1600)  output_proj(1600,1600)               
                rope                cos(1024,32)  sin(1024,32)                             (1024,32)*(1024,32) -> (1024,32)*4       x_even*cos-x_odd*sin   x_even*cos+x_odd*sin                  1e5                      
                                                                     scaled_dot_producted (25,1024,64)@(25,1024,64) -> (25,1024,1024)  q@k                                                       3e9 
                                                                                        (25,1024,1024)@(25,1024,64) -> (25,1024,64)    attn*v                                                    3e9
                                                                                           (1024,1600)@(1600,1600) -> (1024,1600)   multiheads@output_proj                                       5e9
            RMSnorm                              (1600,)                              (1024,1600)*(1600,)  x/rms*self.weight                                                                     4e6
            ffn                     w1(4288,1600) w3(4288,1600) w2(1600,4288)             (1024,1600)@(4288,1600) -> (1024,4288)*2    swilu(x@self.w*(x@self.w3)@self.w2                         3e10
                                                                    (1024,4288)*(1024,4288) -> (1024,4288)     (1024,4288)@(1600,4288)->(1024,1600)                                              4e6+1.5e10
        RMSnorm(ln_final)                        (1600,)                        (1024,1600)*(1600,) -> (1024,1600) x/rms*self.weight                                                             4e6
        Linear(lm_head)                                                                   (1024,1600)@(50257,1600)    x@self.weight                                                              1.6e11 

        memory       6e9 byte
        Args:
            vocab_size (int): _description_
            context_length (int): _description_
            num_layers (int): _description_
            d_model (int): _description_
            num_heads (int): _description_
            d_ff (int): _description_
            rope_theta (float): _description_
        """
        super().__init__()
        self.token_embeddings  = Embedding(vocab_size,d_model,device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,device=device,rope=rope)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSnorm(d_model,device=device)          # 在所有 Transformer Block以后再做一个 RMSNorm 
        self.lm_head = Linear(d_model,vocab_size,device=device)

    def forward(self,x: Int[Tensor, " batch seq_len"]) -> Float[Tensor," batch seq_len vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))



def CrossEntropy(logits: Float[Tensor," batch_size vocab_size"],targets: Int[Tensor," batch_size"]
) -> Float[Tensor,""]:
    # subtact max
    logits_stable = logits - logits.max(dim=-1,keepdim=True).values         # max return tuple(values,indices)
    logits_prob = logits_stable - log(exp(logits_stable).sum(dim=-1,keepdim=True))
    return (-logits_prob[range(targets.size(0)),targets]).mean()

    
class AdamW(torch.optim.Optimizer):
    def __init__(self,params,
                 lr=1e-3,   # learning rate
                 weight_decay=0.01,
                 betas=(0.9,0.95),       # for momentum
                 eps=1e-8,
        ):
        """
        adaw_memory_accounting 
        memory_parameters  = parameters*4   memory_gradients = memory_parameters    memory_optimizer_state = memory_parameters*2     采用float32(4 byte)   mem = total_param*4
                memory(byte)
                                                                                                                         parameter+gradients+optimizer_state                         activation
                token_embedding                                                                                               (vocab_size,d_model)*4*4
                
                Transformer block 
                ‣RMSNorm(s)                                                                                                   (d_model,)*2*4*4                                 (batch_size,context_length,d_model)*2*4               
                ‣Multi-head self-attention sublayer: 𝑄𝐾𝑉 projections, 𝑄𝐾⊤ matrix multiply, softmax,                      (d_model,d_model)*4*4*4                     𝑄𝐾𝑉 projections: (batch_size,num_heads,context_length,d_model/num_heads)*3*4
                weighted sum of values, output projection.                                                                                                𝑄𝐾𝑉 projections: (batch_size,num_heads,context_length,context_length)*4   softmax:   𝑄𝐾𝑉 projections,
                                                                                                                                                           weighted sum of values: (batch_size,num_heads,context_length,d_model/num_heads)*4      
                                                                                                                                                           output projection.: (batch_size,context_length,d_model)*4 
                ‣Position-wise feed-forward (SwiGLU): 𝑊1, 𝑊2, SiLU on the gate branch, element-wise                       (d_ff,d_model)*2*4*4 (d_model,d_ff)*4*4     W1+W2(batch_size,seq_len,d_ff)*2*4 silu(batch_size,seq_len,d_ff)*4
                                                                                                                                                                        element_wise_produnct:(batch_size,seq_len,d_ff)*4
                                                                                                                                                                        W3 (batch_size,seq_len,d_model)*4
                product, 𝑊3
            • final RMSNorm                                                                                                                                                  (batch_size,context_length,d_model)*4  
            • output embedding                                                                                                                                                
            • cross-entropy on logits                                                                                                                                      (batch_size,context_length,vocab_size)*4
          return load_checkpoint(src,model,optimizer)
            Model State Memory = 16 * vocab_size * d_model + num_layers * (64 * d_model^2 + 48 * d_ff * d_model + 16 * d_model) Bytes

            Activation Memory = L * (28 * batch_size * context_length * d_model + 16 * batch_size * context_length * d_ff + 4 * batch_size * num_heads * context_length^2) + 4 * batch_size * context_length * d_model + 4 * batch_size * context_length * vocab_size Bytes
        Args:
            params (_type_): _description_
            lr (_type_, optional): _description_. Defaults to 1e-3.
            betas (tuple, optional): _description_. Defaults to (0.9,0.95).
        """
        defaults = {"lr":lr,"weight_decay":weight_decay,"betas": betas,"eps":eps}
        super().__init__(params,defaults)
    
    def step(self,closure: Optional[Callable]=None):                 #type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weights_decay = group["weight_decay"]
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # for t = 1,... T do 
                state =self.state[p]
                t = state.get("t",1)
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                v = state["v"]
                grad = p.grad.data
                lr_t = lr*(1-beta2**t)**0.5/(1-beta1**t)            #Compute adjustied lr for iteration t
                theta = p.data - lr*weights_decay*p.data                 # apply weight decay independently
                m = beta1*m+(1-beta1)*grad                               # update the first moment estimate
                v = beta2*v+(1-beta2)*(grad.pow(2)) # exp是Tensor的方法                 # update the second moment estimate
                p.data = theta - lr_t*m/(v.sqrt()+eps)                   # apply moment-adjusted weight updates 
                state["t"] = t + 1                                       # increment iteration number
                state["m"] = m
                state["v"] = v                                           # update the moment hyperparam
            
        return loss


def lr_cosine_schedule(t: int,lr_max: float,lr_min: float,T_w: int,T_c: int) -> float:
    """

    Args:
        t (int): current iteration t
        lr_max (float): maximum learning rate
        lr_min (float): minimum learning rate
        T_w (int): the number of warm-up iteration
        T_c (int): the final iteration of cosine annealing 

    Returns:
        float: the learning rate at iteration t
    """
    if t < T_w:
        return t/T_w*lr_max
    elif t >= T_w and t <= T_c:
        return lr_min + (1 + math.cos((t-T_w)/(T_c-T_w)*math.pi))/2*(lr_max-lr_min)
    else:
        return lr_min


def gradient_clipping(parameters: Iterable[nn.Parameter],max_l2_norm: float,eps = 1e-6):
    total_grad = []
    for p in parameters:
        if p.grad is not None:
            total_grad.append(p.grad.flatten())       # 原地修改
    total_norm_sq = torch.cat(total_grad).norm().item()
    if total_norm_sq > max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad = p.grad*(max_l2_norm/(total_norm_sq+eps))
            
             
        
def DataLoading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor,torch.Tensor]:
    start_idx = np.random.randint(0,len(dataset)-context_length,batch_size)
    offsets = np.arange(context_length)
    indices = start_idx[:, None]+offsets[None, :]
    splseq_array = dataset[indices]
    tar_array = dataset[indices + 1]
    splseq = torch.tensor(splseq_array,device=device,dtype=torch.long)
    tar = torch.tensor(tar_array,device=device,dtype=torch.long)
    return (splseq,tar)

    
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),        #模型参数
        "optimizer": optimizer.state_dict(),       #优化器状态
        "iteration": iteration,                  #迭代数
    }
    torch.save(checkpoint,out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

    
def decoding (tokenizer: Tokenizer,model: TransformLM,prompt: str,max_num: int,tem: float,top_p: float) -> str:
    device  = next(model.parameters()).device
    tokens = torch.tensor(tokenizer.encode(prompt),device=device,dtype = torch.long)
    tokens = rearrange(tokens,"seq_len -> 1 seq_len")
    endID = tokenizer.byte_to_id[b"<|endoftext|>"]
    for i in range(max_num):
        logits = model(tokens)
        # decode trick
        logits = logits/tem
        prob = softmax(logits[:,-1,:],-1).squeeze(0)           # prob   (vocab,)
        prob_sorted,indices_sorted = torch.sort(prob,descending=True)
        prob_cumsum = prob_sorted.cumsum(dim=-1)
        k = torch.searchsorted(prob_cumsum,top_p)
        prob_top_p = prob_sorted*(torch.arange(prob.size(-1)) <= k)
        prob_top_p_twodim = rearrange(prob_top_p," vocab_size -> 1 vocab_size")
        next_token = indices_sorted[torch.multinomial(prob_top_p_twodim,1).item()]             # multinomial能够实现归一化   # type: ignore
        if next_token == endID:
            break
        next_token = next_token.reshape(1,1)
        tokens = torch.cat([tokens,next_token],dim=-1)
    
    tokens = rearrange(tokens,"1 seq_len -> seq_len").tolist()
    return tokenizer.decode(tokens)


        
    
    