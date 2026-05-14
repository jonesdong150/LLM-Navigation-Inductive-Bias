import os
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------
# Cache (avoid re-loading on every query)
# ---------
_CACHED: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def _load_model(model_path: str):
    """
    Load tokenizer & model once per model_path and cache them.
    """
    if model_path in _CACHED:
        return _CACHED[model_path]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",          # NEW: replaces deprecated torch_dtype
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    _CACHED[model_path] = (tokenizer, model)
    return tokenizer, model


def run_llm(
    prompt: str,
    *,
    model_path: Optional[str] = None,
    enable_thinking: bool = True,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    do_sample: bool = False,
):
    if model_path is None:
        model_path = os.environ.get(
            "SHIYAN_MODEL_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "Qwen3-0.6B", "Qwen", "Qwen3-0.6B"),
        )

    tokenizer, model = _load_model(model_path)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated_ids = model.generate(**gen_kwargs)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    index = 0
    if enable_thinking:
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

    thinking = (
        tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        if enable_thinking else ""
    )
    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    return {
        "thinking": thinking,
        "answer": answer,
        "raw_output": "",
    }




if __name__ == "__main__":
    # Optional quick manual test (won't run on import)
    out = run_llm("Please introduce large language models.")
    print("THINKING:\n", out["thinking"])
    print("ANSWER:\n", out["answer"])
