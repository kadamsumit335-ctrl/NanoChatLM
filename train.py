
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import tiktoken
import numpy as np
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

class Config:
    vocab_size   = 50257
    block_size   = 128
    n_layer      = 6
    n_head       = 6
    n_embd       = 384
    n_kv_heads   = 2
    dropout      = 0.1
    batch_size   = 32
    max_iters    = 5000
    eval_every   = 500
    lr           = 3e-4
    weight_decay = 0.1
    grad_clip    = 1.0
