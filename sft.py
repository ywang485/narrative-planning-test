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

import json
import logging
import os
import re
from typing import List, Tuple

# Must be set before `import torch`.
# Instructs PyTorch not to reserve the full MPS memory pool upfront.
#os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# Makes MPS ops that are not correctly implemented (e.g. scaled_dot_product_attention
# with causal masks) silently fall back to CPU instead of producing NaN/inf.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer
from util import patch_forward_with_safe_logits, patch_generate_with_safe_logits

# ---------------------------------------------------------------------------
# Logging (initialised early so config loading can emit messages)
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Configuration — config file + env-vars
#
# Values are resolved in this priority order (highest first):
#   1. Environment variable
#   2. JSON config file  (default: sft_config.json, override with CONFIG_FILE=)
#   3. Built-in default
#
# Example config file (sft_config.json):
#   {
#     "dataset_path": "dayo_dataset.jsonl",
#     "eval_dataset_path": "test.jsonl",
#     "eval_interval_epochs": 2
#   }
# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict:
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        log.info("Loaded config file '%s'.", path)
        return cfg
    return {}

_cfg = _load_config(os.getenv("CONFIG_FILE", "sft_config.json"))

def _get(key: str, env_var: str, default):
    """Resolve a config value: env-var > config file > default."""
    env_val = os.getenv(env_var)
    if env_val is not None:
        return env_val
    return _cfg.get(key, default)

MODEL_NAME   = _get("model_name",  "MODEL_NAME",  "Qwen/Qwen2.5-7B-Instruct")
OUTPUT_DIR   = _get("output_dir",  "OUTPUT_DIR",  "./qwen2.5-7b-sft-lora")
DATASET_PATH = _get("dataset_path", "DATASET_PATH", "")

EVAL_DATASET_PATH    = _get("eval_dataset_path",    "EVAL_DATASET_PATH",    "")
EVAL_INTERVAL_EPOCHS = int(_get("eval_interval_epochs", "EVAL_INTERVAL_EPOCHS", 1))
EVAL_GEMINI_MODEL    = _get("eval_gemini_model",    "EVAL_GEMINI_MODEL",    "gemini-2.0-flash")
EVAL_MAX_SAMPLES     = int(_get("eval_max_samples", "EVAL_MAX_SAMPLES",     20))

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
    Load the training dataset.

    If DATASET_PATH points to a JSONL file (e.g. produced by
    build_dayo_dataset.py), every line is parsed as JSON and must contain a
    'text' key with a fully-formatted Qwen2.5 chat string.

    Otherwise the small built-in demonstration dataset is used.
    """
    if DATASET_PATH and os.path.isfile(DATASET_PATH):
        print(f"Loading dataset from '{DATASET_PATH}' …")
        with open(DATASET_PATH, encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(examples)} examples.")
        return Dataset.from_list(examples)

    print("DATASET_PATH not set or file not found — using built-in demo dataset.")
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
# Evaluation — LLM-as-judge (Gemini)
# ---------------------------------------------------------------------------
_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def _extract_question(text: str) -> str:
    m = _USER_RE.search(text)
    return m.group(1).strip() if m else text


def load_eval_examples(path: str, max_samples: int) -> List[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    if max_samples > 0:
        lines = lines[:max_samples]
    return [_extract_question(json.loads(l)["text"]) for l in lines]


def _generate_response(model, tokenizer, question: str, device: str,
                        max_new_tokens: int = 150) -> str:
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


class GeminiJudge:
    """Calls Gemini to decide whether a piece of text is a valid 打油诗."""

    _SYSTEM = (
        "你是评判打油诗的专家。给你一段文字，判断它是否是合格的打油诗。\n"
        "打油诗的标准：多行诗歌结构、押韵、语气幽默风趣、贴近日常生活。\n"
        "先回答 YES 或 NO（大写），然后用一句中文说明原因。"
    )

    def __init__(self, model_name: str) -> None:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is required for evaluation.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name, system_instruction=self._SYSTEM)

    def judge(self, text: str) -> Tuple[bool, str]:
        try:
            resp = self._model.generate_content(f"请判断以下文字是否是打油诗：\n\n{text}")
            verdict = resp.text.strip()
            return verdict.upper().startswith("YES"), verdict
        except Exception as exc:
            log.warning("Gemini judge call failed: %s", exc)
            return False, f"error: {exc}"


class DayoEvalCallback(TrainerCallback):
    """Runs 打油诗 quality evaluation after every `eval_interval` training epochs."""

    def __init__(self, tokenizer, questions: List[str], judge: GeminiJudge,
                 eval_interval: int, device: str) -> None:
        self.tokenizer      = tokenizer
        self.questions      = questions
        self.judge          = judge
        self.eval_interval  = eval_interval
        self.device         = device

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch % self.eval_interval != 0:
            return control

        log.info("── 打油诗 eval  epoch=%d  n=%d ──", epoch, len(self.questions))
        model.eval()
        results = []
        try:
            for q in self.questions:
                answer = _generate_response(model, self.tokenizer, q, self.device)
                passed, verdict = self.judge.judge(answer)
                results.append({
                    "epoch": epoch, "question": q,
                    "answer": answer, "passed": passed, "verdict": verdict,
                })
        finally:
            model.train()

        n_passed  = sum(r["passed"] for r in results)
        pass_rate = n_passed / len(results) if results else 0.0
        log.info("Epoch %d — 打油诗 pass rate: %d/%d (%.1f%%)",
                 epoch, n_passed, len(results), pass_rate * 100)

        # Append per-example results to a file inside OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_path = os.path.join(OUTPUT_DIR, "eval_results.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Eval details appended to '%s'.", log_path)

        # Push scalar metric to wandb when enabled
        if "wandb" in (args.report_to or []):
            try:
                import wandb
                wandb.log({"eval/dayo_pass_rate": pass_rate}, step=state.global_step)
            except Exception as exc:
                log.warning("wandb logging failed: %s", exc)

        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
    model = apply_lora(model)
    if device == "mps":
        model = patch_forward_with_safe_logits(model)   # guards cross-entropy forward pass
        model = patch_generate_with_safe_logits(model)  # guards inference/eval sampling

    dataset = build_dataset()

    # Set up LLM-as-judge eval callback (skipped when EVAL_DATASET_PATH is unset)
    eval_callback = None
    if EVAL_DATASET_PATH and os.path.isfile(EVAL_DATASET_PATH):
        questions = load_eval_examples(EVAL_DATASET_PATH, EVAL_MAX_SAMPLES)
        judge     = GeminiJudge(EVAL_GEMINI_MODEL)
        eval_callback = DayoEvalCallback(
            tokenizer=tokenizer,
            questions=questions,
            judge=judge,
            eval_interval=EVAL_INTERVAL_EPOCHS,
            device=device,
        )
        log.info("Eval enabled: %d questions every %d epoch(s) via %s",
                 len(questions), EVAL_INTERVAL_EPOCHS, EVAL_GEMINI_MODEL)
    else:
        log.info("Eval disabled (set EVAL_DATASET_PATH to enable).")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=100,

        # ── Batch / memory ──────────────────────────────────────────────────
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8

        # ── Optimiser ────────────────────────────────────────────────────────
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # ── SFT-specific ──────────────────────────────────────────────────────
        max_length=512,
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
        callbacks=[eval_callback] if eval_callback else None,
    )

    print("Starting SFT training …")
    trainer.train()

    print(f"\nSaving adapter + tokenizer to '{OUTPUT_DIR}' …")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
