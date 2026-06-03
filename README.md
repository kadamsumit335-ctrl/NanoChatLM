# NanoChatLM

A 28M parameter language model built completely from scratch.

## Architecture
- Rotary Positional Embedding (RoPE)
- RMSNorm (Pre-normalization)
- SwiGLU activation
- Grouped Query Attention (GQA)

<img width="561" height="151" alt="image" src="https://github.com/user-attachments/assets/c9f06bd2-1da0-492f-baf2-a4f5fabff814" />


## Training
- Dataset: OpenAssistant OASST1
- Tokens: 15.7M
- Steps: 5000
- Final val loss: 3.29
- Hardware: Google Colab T4 (free)

## Results
Loss curve: 10.88 → 3.29 over 5000 steps

## Stack
Python · PyTorch · HuggingFace Datasets · Tiktoken
