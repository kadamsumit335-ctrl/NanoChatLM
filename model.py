
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    lr           = 3e-4
    weight_decay = 0.01
    max_iters    = 5000
    grad_clip    = 1.0
    eval_every   = 500

cfg = Config()

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

def precompute_rope(head_dim, seq_len, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos   = torch.arange(seq_len).float()
    freqs = torch.outer(pos, theta)
    return torch.cos(freqs).to(device), torch.sin(freqs).to(device)

def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    x1, x2 = x[..., :D//2], x[..., D//2:]
    return torch.cat([x1*cos[:T] - x2*sin[:T],
                      x1*sin[:T] + x2*cos[:T]], dim=-1)

class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head   = cfg.n_head
        self.n_kv     = cfg.n_kv_heads
        self.head_dim = cfg.n_embd // cfg.n_head
        self.groups   = cfg.n_head // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)
        cos, sin = precompute_rope(self.head_dim, cfg.block_size)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C   = x.shape
        H, Hkv, D = self.n_head, self.n_kv, self.head_dim
        q = self.q_proj(x).view(B, T, H,   D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        att = (q @ k.transpose(-2, -1)) * (D ** -0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden  = int(n_embd * 8 / 3)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.attn  = GQAttention(cfg)
        self.mlp   = SwiGLU(cfg.n_embd)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class NanoChatLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm    = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        x      = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x  = block(x)
        x      = self.norm(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -cfg.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature
            v, _      = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, next_tok], dim=1)
        return idx
