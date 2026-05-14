"""
Llama model inference example using HuggingFace transformers.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/root/autodl-tmp/model/Llama-3.2-1B/LLM-Research/Llama-3.2-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare input
prompt = "Write a short poem about AI."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Decode output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
