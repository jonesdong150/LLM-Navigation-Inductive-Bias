import os
import torch
from typing import Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

_CACHED: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

def _load_model(model_path: str):
    if model_path in _CACHED:
        return _CACHED[model_path]

    abs_path = str(Path(model_path).resolve())
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model path does not exist: {abs_path}")

    print(f"Loading model from: {abs_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        abs_path, 
        trust_remote_code=True,
        local_files_only=True 
    )
    
    # Explicitly set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        abs_path,
        dtype=torch.bfloat16,  # Use dtype parameter
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    model.eval()
    _CACHED[model_path] = (tokenizer, model)
    return tokenizer, model

def run_llm(
    prompt: str,
    *,
    model_path: Optional[str] = None,
    enable_thinking: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    do_sample: bool = False,
):
    if model_path is None:
        model_path = os.environ.get("SHIYAN_MODEL_PATH")

    tokenizer, model = _load_model(model_path)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Base generation config
    gen_kwargs = dict(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    
    # Only add temperature and top_p when sampling is enabled
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        generated_ids = model.generate(**gen_kwargs)

    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:]
    
    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return {
        "thinking": "", 
        "answer": answer,
        "raw_output": answer,
    }