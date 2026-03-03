#!/usr/bin/env python3
"""
Run inference with a base or LoRA fine-tuned Qwen2.5 model.

Usage:
  # Base model only
  python infer.py --model Qwen/Qwen2.5-7B-Instruct

  # Fine-tuned LoRA adapter on top of base model
  python infer.py --model Qwen/Qwen2.5-7B-Instruct --adapter ./qwen2.5-7b-grpo-lora

  # Interactive prompt loop (default)
  python infer.py --model Qwen/Qwen2.5-7B-Instruct --adapter ./qwen2.5-7b-grpo-lora

  # Single prompt, then exit
  python infer.py --model Qwen/Qwen2.5-7B-Instruct --prompt "Explain gradient descent."
"""

import argparse
import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, adapter_path: str | None, device: str):
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    print(f"Loading tokenizer from '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model '{model_name}' ({dtype}) …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    if adapter_path:
        print(f"Loading LoRA adapter from '{adapter_path}' …")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.to(dtype)  # keep dtypes consistent

    model.eval()
    print("Ready.\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, device: str, **gen_kwargs) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
            temperature=gen_kwargs.get("temperature", 0.7),
            top_p=gen_kwargs.get("top_p", 0.9),
            do_sample=True,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 inference script")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--prompt", default=None, help="Single prompt; omit for interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, tokenizer = load_model(args.model, args.adapter, device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.prompt:
        response = generate(model, tokenizer, args.prompt, device, **gen_kwargs)
        print(response)
        return

    # Interactive loop
    print("Enter a prompt (Ctrl-D or blank line to quit):")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        response = generate(model, tokenizer, prompt, device, **gen_kwargs)
        print(f"\n{response}")


if __name__ == "__main__":
    main()
