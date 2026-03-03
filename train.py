#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B with GRPO + LoRA on Apple Silicon (MPS).

Memory requirements:
  - Minimum: 24GB unified memory (fp16 weights alone are ~14GB)
  - Recommended: 32GB+ for comfortable training headroom
  - Alternative: swap MODEL_NAME to "Qwen/Qwen2.5-3B-Instruct" for 16GB Macs

Note: bitsandbytes 4-bit/8-bit quantization is NOT supported on MPS.
Note: the model is loaded in bfloat16 (not float16) — bfloat16's wider
      exponent range prevents inf/nan overflow in Qwen2.5's logits on MPS.

Install dependencies:
  pip install -r requirements.txt
"""

import os

# Must be set before `import torch`.
# Instructs PyTorch not to reserve the full MPS memory pool upfront.
#os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# Makes MPS ops that are not correctly implemented (e.g. scaled_dot_product_attention
# with causal masks) silently fall back to CPU instead of producing NaN/inf.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from util import patch_generate_with_safe_logits

# ---------------------------------------------------------------------------
# Configuration — tweak these to suit your run
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # swap to 3B if memory is tight
OUTPUT_DIR = "./qwen2.5-7b-grpo-lora"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# These target all the linear projections inside each transformer block.
# Qwen2 shares the same projection names as LLaMA-style models.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        print("Apple MPS backend available — using GPU acceleration.")
        return "mps"
    if torch.cuda.is_available():
        print("CUDA available — using NVIDIA GPU.")
        return "cuda"
    print("Warning: no GPU found. CPU training will be very slow.")
    return "cpu"


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Qwen2 may not define pad_token; fall back to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bfloat16 shares float32's exponent range, preventing the overflow that
    # fp16 (max ~65504) produces in Qwen2.5's attention scores and FFN
    # activations — overflow causes inf logits → nan after log_softmax →
    # nan KL and entropy.  PYTORCH_ENABLE_MPS_FALLBACK covers the handful of
    # MPS ops that don't support bf16 natively.
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    print(f"Loading '{model_name}' with dtype={dtype} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        # device_map="auto" relies on accelerate's device placement which does
        # not support MPS. We place the model manually after loading.
    )
    model = model.to(device)
    print("Model loaded.\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------
def apply_lora(model):
    # Capture the base model dtype before PEFT adds anything.
    base_dtype = next(model.parameters()).dtype

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # PEFT initialises adapter weights as float32 (standard nn.Linear defaults),
    # regardless of the base model's dtype.  A bfloat16 base + float32 adapters
    # mix during the forward pass and can produce inf/nan logits on MPS.
    # Re-casting the whole model ensures every parameter shares the same dtype.
    model = model.to(base_dtype)

    model.print_trainable_parameters()
    print()
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset() -> Dataset:
    """
    Minimal demonstration dataset.  Replace with your own.

    Each record must have a 'prompt' key.  The GRPOTrainer will generate
    `num_generations` completions per prompt and score them with your reward
    function(s).
    """
    examples = [
        {"prompt": "Explain gradient descent in simple terms."},
        {"prompt": "What are the key differences between supervised and unsupervised learning?"},
        {"prompt": "Describe how the transformer attention mechanism works."},
        {"prompt": "What is overfitting and how can it be prevented?"},
        {"prompt": "Explain the bias-variance tradeoff."},
        {"prompt": "What is backpropagation and why is it important?"},
        {"prompt": "How does weight decay act as a regularizer?"},
        {"prompt": "What is the purpose of batch normalization?"},
    ]
    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# Reward function(s)
# ---------------------------------------------------------------------------
def reward_length(completions: list[str], **kwargs) -> list[float]:
    """
    Demo reward: a Gaussian bell centred on a target word count.

    Replace or extend this with a meaningful reward signal, for example:
      - a trained reward model
      - rule-based quality / correctness checks
      - tool-call success / code execution results
      - human preference scores

    Args:
        completions: list of generated text strings (one per sample in the
                     rolled-out batch).
        **kwargs: the GRPOTrainer may pass additional fields from the dataset
                  row (e.g. ground-truth answers).

    Returns:
        List of scalar reward values, one per completion.
    """
    target_words = 150
    sigma = 80
    rewards = []
    for text in completions:
        n_words = len(text.split())
        reward = float(
            torch.exp(
                torch.tensor(-((n_words - target_words) ** 2) / (2 * sigma ** 2))
            )
        )
        rewards.append(reward)
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
    model = apply_lora(model)
    if device == "mps":
        model = patch_generate_with_safe_logits(model)

    dataset = build_dataset()

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,

        # ── Batch / memory ──────────────────────────────────────────────────
        # Keep the per-device batch small — a 7B model leaves little headroom.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8

        # ── Optimiser ────────────────────────────────────────────────────────
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,

        # ── GRPO-specific ─────────────────────────────────────────────────────
        # Number of completions sampled per prompt per update step.
        # Lower values save memory; minimum useful value is 2.
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=256,
        temperature=0.9,
        # KL penalty coefficient (β in the GRPO paper).
        # Larger β keeps the policy closer to the reference; 0.0 disables KL.
        beta=0.04,

        # ── Precision ─────────────────────────────────────────────────────────
        # Do NOT enable fp16/bf16 flags here — we handle dtype at model-load
        # time.  Enabling both causes double-casting on MPS and can produce NaNs.
        fp16=False,
        bf16=False,

        # ── MPS compatibility ─────────────────────────────────────────────────
        # pin_memory uses CUDA-pinned memory which is unavailable on MPS.
        dataloader_pin_memory=False,
        # vLLM is a CUDA-only inference back-end; disable it explicitly.
        use_vllm=False,

        # ── Logging / checkpointing ───────────────────────────────────────────
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="wandb",         # set to "wandb" or "tensorboard" if desired
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,   # 'tokenizer=' also accepted in older TRL
        reward_funcs=reward_length,   # pass a list for multiple reward signals
    )

    print("Starting GRPO training …")
    trainer.train()

    print(f"\nSaving adapter + tokenizer to '{OUTPUT_DIR}' …")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
