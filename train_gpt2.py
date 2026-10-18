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


# ----------------------- Model -----------------------------
@dataclass  # optimization 5 (use nice numbers -- multiples/powers of 2)
class GPT2Config:
    block_size: int = 256  # context window
    # gpt-2 uses vocab_size of 50257 tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    # we round it up for efficiency
    vocab_size: int = 50304
    n_embd: int = 16
    n_layer: int = 4
    n_head: int = 4  # no. of heads in a multi-head attention block


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
        out = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True)  # optimization 4

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
                "wte": nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
                "wpe": nn.Embedding(config.block_size, config.n_embd),  # positional embedding
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme for GPT-2
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Input sequence is longer than block size"

        # obtain position and token embeddings
        pos = torch.arange(start=0, end=T, dtype=torch.long, device=idx.device)  # (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd) -- positional embedding
        tok_emb = self.transformer.wte(idx)  # (B,T,n_embd) -- token embedding
        x = pos_emb + tok_emb
        # pass embedded input through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # final layer norm and fully connected layer
        x = self.transformer.ln_f(x)  # (B,T,n_embd)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=targets.view(-1))

        return logits, loss


# ----------------------- Data -----------------------------
class DataLoaderLite:
    def __init__(self, B, T, split):

        # obtain micro-batch size and max sequence length (context length)
        self.B = B
        self.T = T

        # load data and encode into tokens
        # NOTE: let this happen on CPU
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


# ----------------------- Training loop -----------------------------
# device
if torch.cuda.is_available():
    device = "cuda"
    device_type = "cuda"
    torch.set_float32_matmul_precision("high")  # optimization 1 (only for cuda)
elif torch.backends.mps.is_available():
    device = "mps"
    device_type = "mps"
else:
    device = "cpu"
    device_type = "cpu"
print(f"Using device: {device}")

# hyperparameters
lr = 3e-4
batch_size = 8
num_max_steps = 10000
eval_steps = 1000  # every few steps run validation and generation
val_loss_steps = 20  # to calculate average val loss over 20 validation batches


# data loader
train_dataloader = DataLoaderLite(B=batch_size, T=GPT2Config.block_size, split="train")
val_dataloader = DataLoaderLite(B=batch_size, T=GPT2Config.block_size, split="val")

# model
model = GPT(GPT2Config())
model = model.to(device)
model = torch.compile(model)  # optimization 3


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

train_loss = []
val_loss = []

for step in range(num_max_steps):
    # training stage
    model.train()
    x, y = train_dataloader.next_batch()
    x, y = x.to(device), y.to(device)  # move tokens to device (GPU)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # optimization 2 (Bfloat16)
        logits, loss = model(idx=x, targets=y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % eval_steps == 0:
        train_loss.append(loss.item())

    # validation stage
    if step % eval_steps == 0:
        model.eval()
        val_loss_accum = 0  # avg val loss
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_dataloader.next_batch()
                x, y = x.to(device), y.to(device)  # move tokens to device (GPU)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # optimization 2 (Bfloat16)
                    _, loss = model(idx=x, targets=y)
                    val_loss_accum += loss.item() / val_loss_steps
        val_loss.append(val_loss_accum)

    # generate
    # source: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
    if (step % eval_steps == 0) or (step == num_max_steps - 1):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"{decoded}")
            print("--------------------------")

# plot train and val loss
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(train_loss, label="train loss")
ax.plot(val_loss, label="val loss")
ax.legend()
plt.show()

code.interact(local=locals())
