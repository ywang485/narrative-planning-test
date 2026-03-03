#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B with SFT + LoRA on Apple Silicon (MPS).

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
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from util import patch_generate_with_safe_logits

# ---------------------------------------------------------------------------
# Configuration — tweak these to suit your run
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # swap to 3B if memory is tight
OUTPUT_DIR = "./qwen2.5-7b-sft-lora"

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

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

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

    Each record must have a 'text' key containing the fully-formatted training
    example (e.g. a chat template string).  SFTTrainer trains the model to
    predict every token in 'text' by default; set dataset_text_field or use
    a formatting_func for more control.
    """
    examples = [
        {
            "text": (
                "<|im_start|>user\nExplain gradient descent in simple terms."
                "<|im_end|>\n<|im_start|>assistant\n"
                "Gradient descent is an optimisation algorithm that iteratively "
                "adjusts a model's parameters in the direction that reduces the "
                "loss the most, like walking downhill by always stepping in the "
                "steepest downward direction.<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nWhat are the key differences between supervised "
                "and unsupervised learning?<|im_end|>\n<|im_start|>assistant\n"
                "Supervised learning trains on labelled examples to predict outputs, "
                "while unsupervised learning finds structure in unlabelled data such "
                "as clusters or latent representations.<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nDescribe how the transformer attention mechanism works."
                "<|im_end|>\n<|im_start|>assistant\n"
                "Attention computes a weighted sum of value vectors, where the weights "
                "come from the dot-product similarity between query and key vectors, "
                "scaled and normalised with softmax.<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nWhat is overfitting and how can it be prevented?"
                "<|im_end|>\n<|im_start|>assistant\n"
                "Overfitting is when a model memorises training data and fails to "
                "generalise. It can be prevented with regularisation, dropout, early "
                "stopping, or gathering more training data.<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nExplain the bias-variance tradeoff."
                "<|im_end|>\n<|im_start|>assistant\n"
                "Bias measures how far average predictions are from the truth; variance "
                "measures how much predictions change across datasets. Reducing one "
                "typically increases the other, so good models balance both.<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nWhat is backpropagation and why is it important?"
                "<|im_end|>\n<|im_start|>assistant\n"
                "Backpropagation efficiently computes gradients of the loss with respect "
                "to every parameter by applying the chain rule backwards through the "
                "computation graph, making large-scale neural network training feasible."
                "<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nHow does weight decay act as a regularizer?"
                "<|im_end|>\n<|im_start|>assistant\n"
                "Weight decay adds a penalty proportional to the squared magnitude of "
                "the weights to the loss, discouraging large parameter values and "
                "reducing overfitting.<|im_end|>"
            )
        },
        {
            "text": (
                "<|im_start|>user\nWhat is the purpose of batch normalization?"
                "<|im_end|>\n<|im_start|>assistant\n"
                "Batch normalisation standardises layer inputs to have zero mean and "
                "unit variance within each mini-batch, stabilising and accelerating "
                "training by reducing internal covariate shift.<|im_end|>"
            )
        },
    ]
    return Dataset.from_list(examples)


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

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,

        # ── Batch / memory ──────────────────────────────────────────────────
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8

        # ── Optimiser ────────────────────────────────────────────────────────
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # ── SFT-specific ──────────────────────────────────────────────────────
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,   # set True to pack short examples into one sequence

        # ── Precision ─────────────────────────────────────────────────────────
        # Do NOT enable fp16/bf16 flags here — we handle dtype at model-load
        # time.  Enabling both causes double-casting on MPS and can produce NaNs.
        fp16=False,
        bf16=False,

        # ── MPS compatibility ─────────────────────────────────────────────────
        # pin_memory uses CUDA-pinned memory which is unavailable on MPS.
        dataloader_pin_memory=False,

        # ── Logging / checkpointing ───────────────────────────────────────────
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training …")
    trainer.train()

    print(f"\nSaving adapter + tokenizer to '{OUTPUT_DIR}' …")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
