import torch
import tiktoken
from model import NanoChatLM, cfg, device

enc   = tiktoken.get_encoding("gpt2")
model = NanoChatLM(cfg).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

def generate(prompt, max_tokens=150, temperature=0.8, top_k=40):
    tokens = enc.encode(prompt)
    idx    = torch.tensor([tokens], dtype=torch.long).to(device)
    out    = model.generate(idx, max_tokens, temperature, top_k)
    return enc.decode(out[0].tolist())

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    print(generate(prompt))
