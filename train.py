import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import tiktoken
import numpy as np
import os
from model import NanoChatLM, Config, cfg, device

# ── Dataset ───────────────────────────────────
ds = load_dataset("OpenAssistant/oasst1")

def flatten_oasst(split):
    text = ""
    for row in split:
        if row["text"]:
            text += row["text"].strip() + "\n"
    return text

train_text = flatten_oasst(ds["train"])
val_text   = flatten_oasst(ds["validation"])

# ── Tokenize ──────────────────────────────────
enc        = tiktoken.get_encoding("gpt2")
train_ids  = enc.encode(train_text, allowed_special={"<|endoftext|>"})
val_ids    = enc.encode(val_text,   allowed_special={"<|endoftext|>"})
train_data = torch.tensor(train_ids, dtype=torch.long)
val_data   = torch.tensor(val_ids,   dtype=torch.long)
print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

# ── Batch ─────────────────────────────────────
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix   = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x    = torch.stack([data[i:i+cfg.block_size]     for i in ix])
    y    = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def eval_loss(model):
    model.eval()
    losses = [model(*get_batch("val"))[1].item() for _ in range(20)]
    model.train()
    return np.mean(losses)

# ── Train ─────────────────────────────────────
model     = NanoChatLM(cfg).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr,
    weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.max_iters, eta_min=cfg.lr/10
)

model.train()
for step in range(cfg.max_iters):
    x, y    = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    scheduler.step()
    if step % cfg.eval_every == 0:
        vl = eval_loss(model)
        print(f"step {step:5d} | train {loss.item():.4f} | val {vl:.4f}")

# ── Save ──────────────────────────────────────
torch.save(model.state_dict(), "model.pt")
print("Training complete. Saved model.pt ")
