import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen1.5-0.5B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Tokenizer loaded")
except Exception as e:
    print(e)
