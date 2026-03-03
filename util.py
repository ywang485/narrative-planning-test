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
