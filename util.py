"""Shared utilities for train.py and infer.py."""

import functools

import torch
from transformers import LogitsProcessor, LogitsProcessorList


class _NaNSafeLogits(LogitsProcessor):
    """Replace any inf/nan in logits before they reach torch.multinomial.

    MPS scaled_dot_product_attention can still emit NaN despite the fallback
    flag in edge cases (e.g. all-padding positions, extreme activations).
    This processor is the last line of defense before the sampling step.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # nan  → -inf  (zero probability after softmax)
        # +inf → large-but-finite (preserves relative ordering)
        return torch.nan_to_num(scores, nan=float("-inf"), posinf=1e4, neginf=float("-inf"))


def patch_forward_with_safe_logits(model):
    """Register a forward hook on lm_head to sanitize NaN/inf logits.

    patch_generate_with_safe_logits only guards the sampling step.
    The GRPO trainer's KL and entropy are computed from log_softmax over the
    raw logits produced by model(...) forward calls, which bypass the logits
    processor entirely.  A single NaN logit in any row makes logsumexp return
    NaN for that row, poisoning the whole loss.

    We use nan → -1e4 (not -inf) so that an all-NaN row stays finite after
    log_softmax.  -inf would leave log_softmax(-inf, -inf, ...) = NaN again.
    """
    def _hook(_module, _input, output):
        return torch.nan_to_num(output, nan=-1e4, posinf=1e4, neginf=-1e4)

    # Unwrap PEFT layers to reach the underlying Qwen2ForCausalLM.lm_head
    inner = getattr(model, "base_model", model)
    inner = getattr(inner, "model", inner)
    if hasattr(inner, "lm_head"):
        inner.lm_head.register_forward_hook(_hook)
    return model


def patch_generate_with_safe_logits(model):
    """Wrap model.generate so _NaNSafeLogits is always the first processor."""
    _processor = _NaNSafeLogits()
    _orig = model.generate

    @functools.wraps(_orig)
    def _safe_generate(*args, **kwargs):
        lp = kwargs.pop("logits_processor", None) or LogitsProcessorList()
        lp.insert(0, _processor)
        return _orig(*args, logits_processor=lp, **kwargs)

    model.generate = _safe_generate
    return model
