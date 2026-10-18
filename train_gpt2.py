"""
Reference: Andrej Karpathy's lecture and resources on GPT-2 here -- https://www.youtube.com/watch?v=l8pRSuU81PU&t=7639s
"""

import code
from dataclasses import dataclass

import torch
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F

import tiktoken

torch.manual_seed(1266)

# Check for CUDA (NVIDIA), then MPS (Apple Silicon), otherwise default to CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# ----------------------- Model -----------------------------
@dataclass
class GPT2Config:
    block_size: int = 256  # context window
    vocab_size: int = 65
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # k,q,v for all heads in a multi-head attn setup
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # skip connection projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # B=batch_size, T=block_size, C=n_embd

        # calculate q,k, and v for x in one pass
        qkv = self.c_attn(x)  # (B,T,3*C)

        # separate out the q,k and v matrices
        q, k, v = qkv.split(self.n_embd, dim=2)  # (B,T,C)

        # reshape these matrices to make "head" a batch dim
        # nh = no. of heads, hs = head size (in GPT2 its = n_embd//n_head)
        # .transpose(1,2) because torch self-attn func expects that
        # (B, T, nh, hs) -> for each word, you have all heads
        # (B, nh, T, hs) -> for each head, you have all words (Torch expects this)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,nh,T,hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,nh,T,hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,nh,T,hs)

        # flash attention
        # F.scaled_dot_product_attention returns same shape as "query" i.e. (B,T,nh,hs)
        out = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True)

        # remove "head" from batched dim to get output in (B,T,C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # project skip connection back
        out = self.c_proj(out)
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Emdedding(config.vocab_size, config.n_embd),  # token embedding
                "wpe": nn.Embedding(config.block_size, config.n_embd),  # positional embedding
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, "Input sequence is longer than block size"
        # obtain position and token embeddings
        pos = torch.arange(start=0, end=T, dtype=torch.long, device=idx.device)  # (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd) -- positional embedding
        tok_emb = self.transformer.wte(idx)  # (B,T,n_embd) -- token embedding
        x = pos_emb + tok_emb
        # pass embedded input through transformer blocks
        for block in self.trasnformer.h:
            x = block(x)
        # final layer norm and fully connected layer
        x = self.trasnformer.ln_f(x)  # (B,T,n_embd)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits


# ----------------------- Data -----------------------------
class DataLoaderLite:
    def __init__(self, B, T, split):

        # obtain micro-batch size and max sequence length (context length)
        self.B = B
        self.T = T

        # load data and encode into tokens
        enc = tiktoken.get_encoding("gpt2")
        with open("./nanogpt_data/input.txt", "r", encoding="utf-8") as f:
            text = f.read()
            tokens = enc.encode(text)
        n = int(0.9 * len(tokens))

        # split
        assert split in ["train", "val"], "Invalid split"
        self.tokens = torch.tensor(tokens[:n]) if split == "train" else torch.tensor(tokens[n:])

        # current position
        self.current_position = 0

    def next_batch(self):
        # obtain all tokens needed for one forward pass --> B*T + 1
        buf = self.tokens[self.current_position : self.current_position + (self.B * self.T + 1)]

        # prepare input and targets
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        # update current_position
        self.current_position += self.B * self.T

        # reset current_position if loading next batch would go out of bounds
        if self.current_position + (self.B * self.T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


B = 4
T = GPT2Config.block_size
train_dataloader = DataLoaderLite(B, T, "train")
code.interact(local=locals())


# ----------------------- Training loop -----------------------------
