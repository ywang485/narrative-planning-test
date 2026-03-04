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
import re

# Must be set before `import torch`.
# Instructs PyTorch not to reserve the full MPS memory pool upfront.
#os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# Makes MPS ops that are not correctly implemented (e.g. scaled_dot_product_attention
# with causal masks) silently fall back to CPU instead of producing NaN/inf.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import google.generativeai as genai
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from util import patch_forward_with_safe_logits, patch_generate_with_safe_logits

# ---------------------------------------------------------------------------
# Configuration — tweak these to suit your run
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # swap to 3B if memory is tight
OUTPUT_DIR = "./qwen2.5-7b-grpo-lora"

# Optional: path to a prior checkpoint to use as the starting point for GRPO.
#   - LoRA adapter checkpoint (contains adapter_config.json): the adapter is
#     loaded on top of MODEL_NAME and training continues from those weights.
#   - Full / merged model directory: loaded directly as the base model, then a
#     fresh LoRA adapter is applied before GRPO training starts.
# Leave as "" (or unset CHECKPOINT_PATH env var) to start from MODEL_NAME.
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "")

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


def load_from_checkpoint(checkpoint_path: str, base_model_name: str, device: str):
    """Load a model from a prior checkpoint for use as the GRPO starting point.

    Two checkpoint formats are supported:
      - LoRA adapter directory (contains adapter_config.json): the base model
        is loaded from `base_model_name` and the adapter weights are applied on
        top.  The returned model already has LoRA — do NOT call apply_lora().
      - Full / merged model directory: loaded directly as the base model.
        Call apply_lora() on the returned model as usual.

    Returns:
        (model, tokenizer, lora_already_applied: bool)
    """
    from peft import PeftModel

    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    # Prefer tokenizer from the checkpoint; fall back to the base model.
    tok_source = checkpoint_path if os.path.exists(
        os.path.join(checkpoint_path, "tokenizer_config.json")
    ) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"Checkpoint is a LoRA adapter — loading base '{base_model_name}' …")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        print(f"Attaching adapter weights from '{checkpoint_path}' …")
        model = PeftModel.from_pretrained(base, checkpoint_path).to(dtype)
        print("Checkpoint loaded (LoRA adapter).\n")
        return model, tokenizer, True
    else:
        print(f"Checkpoint is a full model — loading from '{checkpoint_path}' …")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        print("Checkpoint loaded (full model).\n")
        return model, tokenizer, False


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
# Gemini judge helper
# ---------------------------------------------------------------------------
_gemini_model = None

def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the LLM-as-judge reward."
            )
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_model


def _gemini_score(prompt: str, completion: str) -> float:
    """Ask Gemini to rate how well `completion` answers `prompt` on 0–10.
    Returns a float in [0.0, 1.0]. Falls back to 0.0 on any error."""
    judge_prompt = (
        "You are an expert evaluator. Rate how well the following answer addresses "
        "the question. Reply with a single integer from 0 (completely wrong or "
        "irrelevant) to 10 (perfect, accurate, and complete).\n\n"
        f"Question: {prompt}\n\n"
        f"Answer: {completion}\n\n"
        "Score (0-10):"
    )
    try:
        model = _get_gemini_model()
        response = model.generate_content(judge_prompt)
        text = response.text.strip()
        # Extract first integer found in the response
        match = re.search(r"\b(\d{1,2})\b", text)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 10) / 10.0
    except Exception as e:
        print(f"[gemini_score] Error: {e}")
    return 0.0


def _conciseness_score(text: str, ideal: int = 80, max_words: int = 250) -> float:
    """Returns 1.0 for ≤ `ideal` words, linearly decays to 0.0 at `max_words`."""
    n = len(text.split())
    if n <= ideal:
        return 1.0
    if n >= max_words:
        return 0.0
    return 1.0 - (n - ideal) / (max_words - ideal)


# ---------------------------------------------------------------------------
# Reward function(s)
# ---------------------------------------------------------------------------
def reward_llm_judge(completions: list[str], **kwargs) -> list[float]:
    """
    Combined reward: LLM-as-judge quality score (via Google Gemini) + conciseness.

    Requires GEMINI_API_KEY (or GOOGLE_API_KEY) to be set in the environment.

    Score breakdown:
      - 70 % — Gemini rates how well the completion answers the question (0–10 → 0–1)
      - 30 % — Conciseness: full marks up to 80 words, linearly penalised up to 250

    Args:
        completions: list of generated text strings (one per sample in the batch).
        **kwargs: GRPOTrainer passes `prompts` (list[str]) aligned with completions.

    Returns:
        List of scalar reward values in [0.0, 1.0], one per completion.
    """
    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []
    for prompt, completion in zip(prompts, completions):
        llm = _gemini_score(prompt, completion)
        concise = _conciseness_score(completion)
        rewards.append(0.7 * llm + 0.3 * concise)
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()

    if CHECKPOINT_PATH:
        model, tokenizer, lora_applied = load_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)
    else:
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
        lora_applied = False

    if not lora_applied:
        model = apply_lora(model)

    if device == "mps":
        model = patch_forward_with_safe_logits(model)   # guards KL/entropy forward pass
        model = patch_generate_with_safe_logits(model)  # guards sampling step

    dataset = build_dataset()

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=50,

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
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=256,
        temperature=0.9,
        # KL penalty coefficient (β in the GRPO paper).
        # Larger β keeps the policy closer to the reference; 0.0 disables KL.
        beta=0.2,

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
        save_steps=5,
        save_total_limit=2,
        report_to="wandb",         # set to "wandb" or "tensorboard" if desired
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,   # 'tokenizer=' also accepted in older TRL
        reward_funcs=reward_llm_judge,  # LLM-as-judge (Gemini) + conciseness
    )

    print("Starting GRPO training …")
    trainer.train()

    print(f"\nSaving adapter + tokenizer to '{OUTPUT_DIR}' …")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
