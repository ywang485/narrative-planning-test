# Reward Function Design to Avoid Catastrophic Forgetting

## Context

The current GRPO setup trains a Qwen2.5-7B LoRA adapter with:
- **Reward:** `0.7 × gemini_quality + 0.3 × conciseness` (both in [0, 1])
- **KL penalty:** `β=0.04` (very soft; only weakly constrains drift from reference)
- **Reference model:** base Qwen2.5-7B-Instruct (or an SFT checkpoint)
- **Dataset:** 8 static prompts, all in the ML/DL domain

Catastrophic forgetting manifests in two ways here:
1. **GRPO reward hacking** — the policy finds outputs that score high on Gemini/conciseness but lose general capabilities (e.g., starts producing terse, keyword-stuffed answers)
2. **Distribution collapse** — the policy diverges from the reference, erasing SFT-learned behavior

---

## Recommendations (no single "correct" answer — these are layered options)

### Layer 1 — Strengthen the KL anchor (cheapest, do this first)

The existing `beta=0.04` is very permissive. The KL penalty is already built into TRL's GRPO loss; simply increasing it reduces forgetting without any reward function changes.

**In `grpo_config.json`:**
```json
{ "beta": 0.1 }   // conservative; try 0.05–0.2 range
```

Trade-off: higher β slows reward improvement; lower β risks forgetting.

---

### Layer 2 — Add a behavior-preservation component to the reward

Modify `reward_llm_judge` in `train.py` to add a third reward signal that checks whether the model's output is still "in distribution" relative to reference behavior.

**Option A — Reference perplexity penalty (strongest)**

Score how surprised the *reference* (SFT) model is by the completion. Low reference perplexity ⟹ behavior is still close to what SFT taught.

```python
# Requires loading a frozen reference model alongside the training model.
# Pass it as a closure or module-level singleton.
def _reference_perplexity_score(prompt: str, completion: str, ref_model, tokenizer) -> float:
    """Returns 1.0 when completion is high-likelihood under the reference model,
    decays toward 0.0 for high-perplexity outputs."""
    ids = tokenizer(prompt + completion, return_tensors="pt").input_ids.to(ref_model.device)
    with torch.no_grad():
        loss = ref_model(ids, labels=ids).loss   # cross-entropy per token
    # loss ≈ log-perplexity; scale to [0,1] via an exponential
    return float(torch.exp(-loss * 0.5).clamp(0.0, 1.0))
```

New weighting (example):
```python
rewards.append(0.6 * llm + 0.2 * concise + 0.2 * ref_perplexity)
```

Memory cost: one extra frozen model in VRAM (~14 GB for 7B). On 32 GB+ machines this is feasible. On 24 GB it requires offloading the reference to CPU and computing it on CPU.

**Option B — Format/capability regression check (lightweight)**

Add a fast, rule-based check that the completion passes basic quality gates that an SFT model should always satisfy: non-empty, uses sentences, no repetition loops.

```python
def _sanity_score(text: str) -> float:
    """Penalizes degenerate outputs: empty, single-token repetition, garbled text."""
    if len(text.strip()) < 10:
        return 0.0
    words = text.split()
    if len(words) >= 4:
        # Check for repetition loop (common forgetting symptom)
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if len(set(trigrams)) / len(trigrams) < 0.5:
            return 0.0
    return 1.0
```

Use it as a **multiplier** (not an additive term), so any degenerate output gets zeroed out:
```python
rewards.append(_sanity_score(completion) * (0.7 * llm + 0.3 * concise))
```

**Option C — Replay SFT prompts in the GRPO dataset**

Add held-out SFT question/answer pairs to the GRPO training set. For these examples, compute an **exact-match or ROUGE similarity** reward against the SFT reference answer, not a Gemini score. This directly pressures the policy to remember SFT-trained answers.

```python
# In _build_dataset(), tag some examples with a reference answer:
{"prompt": "...", "reference": "..."}   # replay examples
{"prompt": "..."}                        # normal GRPO examples

# In reward_llm_judge(), branch on the presence of "reference":
if reference := kwargs.get("references", [None]*len(completions))[i]:
    llm = _rouge_score(completion, reference)   # no Gemini call
```

---

### Layer 3 — Reward normalization and clipping

GRPO is sensitive to reward scale variance. Highly variable rewards cause large, noisy gradient steps that destabilize the policy. Normalize per-batch:

```python
def reward_llm_judge(completions, **kwargs):
    ...
    raw = [0.7 * llm + 0.3 * concise for llm, concise in scores]
    mean, std = statistics.mean(raw), statistics.stdev(raw) if len(raw) > 1 else 1.0
    return [(r - mean) / (std + 1e-8) for r in raw]
```

TRL's GRPOTrainer normalizes rewards internally by default (controlled by `normalize_rewards=True` in GRPOConfig — check your TRL version). If it's already on, skip this.

---

## Recommended implementation order

1. **Increase `beta` to 0.1** in `grpo_config.json` — zero code change, immediate effect
2. **Add `_sanity_score` as a multiplier** — lightweight, catches degenerate outputs
3. **Add reward normalization** — stabilizes training dynamics
4. **(Optional) Add reference perplexity component** — strongest forgetting protection, but memory-intensive

## Files to modify

- `train.py` — `_conciseness_score`, `reward_llm_judge`, `_build_dataset`
- `grpo_config.json` / `grpo_config.example.json` — `beta`, any new thresholds

## Verification

- Monitor KL divergence in W&B logs — should stay below ~0.5 nats during training
- Run `infer.py` on held-out prompts before and after GRPO to compare output quality/format
- Watch for repetition loops and empty outputs in completions logged during training
