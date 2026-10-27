"""
Evaluation and benchmarking for pruning experiments.

Provides tools to measure:
- Distribution divergence (KL, JS) between original and pruned model outputs
- Top-k token overlap between original and pruned predictions
- Perplexity (output quality proxy, kept for reference)
- Inference latency (wall-clock time per forward pass)
- Memory usage (process RSS)
- Parameter counts (total, trainable, non-zero)

Core philosophy: a good pruning preserves the model's **output distribution**.
So we compare the pruned model's token probabilities against the **original
model's probabilities**, treating the original as the reference distribution.
This is more informative than comparing each model independently against
ground-truth text.
"""

import statistics
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .utils import (
    count_parameters,
    count_nonzero_parameters,
    model_size_mb,
    get_process_memory_mb,
    get_device,
)


# ===========================================================================
# Distribution-divergence metrics (original vs pruned)
# ===========================================================================

@torch.no_grad()
def compute_distribution_divergence(
    original_model: PreTrainedModel,
    pruned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 256,
    stride: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compare the pruned model's output distribution to the original's.

    For each token position we collect the full vocabulary probability
    distribution from both models and compute:

    - **kl_divergence** (nats): D_KL(original || pruned).
      How many extra nats needed to encode the original's distribution
      using the pruned model's distribution.  0 = identical.

    - **js_divergence**: Jensen-Shannon divergence.
      Symmetric, bounded [0, ln 2].  More robust than KL.

    - **top1_agreement**: Fraction of positions where both models agree
      on the single most likely next token.  1.0 = perfect.

    - **top5_overlap**: Average intersection divided by 5 between the top-5
      predictions of each model.  1.0 = perfect.

    Args:
        original_model: Un-pruned baseline model.
        pruned_model: Pruned model to evaluate.
        tokenizer: Shared tokenizer.
        texts: Evaluation texts.
        max_length: Max token window length.
        stride: Sliding-window stride.
        device: Torch device.

    Returns:
        Dict with: kl_divergence, js_divergence, top1_agreement,
        top5_overlap, total_tokens.
    """
    if device is None:
        device = get_device()

    if not 0 < stride < max_length:
        raise ValueError("Require 0 < stride < max_length for continuous next-token coverage.")

    original_model.eval()
    pruned_model.eval()

    total_kl = 0.0
    total_js = 0.0
    total_top1_match = 0
    total_top5_overlap = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Comparing distributions", unit="text"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = enc.input_ids.to(device)
        seq_len = input_ids.size(1)

        previous_end = 0
        for start in range(0, seq_len, stride):
            end = min(start + max_length, seq_len)
            if end - start < 2:
                break

            window = input_ids[:, start:end]

            # -- Get logits from both models --
            orig_logits = original_model(window).logits     # (1, seq, vocab)
            pruned_logits = pruned_model(window).logits

            # -- Convert to probability distributions --
            orig_log_probs = torch.log_softmax(orig_logits.float(), dim=-1)
            pruned_log_probs = torch.log_softmax(pruned_logits.float(), dim=-1)
            orig_probs = orig_log_probs.exp()
            pruned_probs = pruned_log_probs.exp()

            # -- KL divergence: D_KL(orig || pruned) per token --
            log_ratio = orig_log_probs - pruned_log_probs
            kl_per_token = (orig_probs * log_ratio).sum(dim=-1)       # (1, seq)

            # -- JS divergence --
            log_mixture = torch.logaddexp(orig_log_probs, pruned_log_probs) - 0.6931471805599453
            js1 = (orig_probs * (orig_log_probs - log_mixture)).sum(dim=-1)
            js2 = (pruned_probs * (pruned_log_probs - log_mixture)).sum(dim=-1)
            js_per_token = 0.5 * js1 + 0.5 * js2                      # (1, seq)

            # -- Top-1 agreement --
            orig_top1 = orig_logits.argmax(dim=-1)                    # (1, seq)
            pruned_top1 = pruned_logits.argmax(dim=-1)
            top1_match = (orig_top1 == pruned_top1).float()           # (1, seq)

            # -- Top-5 Jaccard overlap --
            _, orig_top5 = orig_logits.topk(5, dim=-1)               # (1, seq, 5)
            _, pruned_top5 = pruned_logits.topk(5, dim=-1)
            # Count intersection per position
            orig_set = orig_top5.unsqueeze(-1)                       # (1, seq, 5, 1)
            pruned_set = pruned_top5.unsqueeze(-2)                   # (1, seq, 1, 5)
            intersection = (orig_set == pruned_set).any(dim=-1).sum(dim=-1).float()
            top5_overlap = intersection / 5.0                        # (1, seq)

            # Score each target once. Position i predicts target i+1;
            # exclude the final output position, not the first one.
            first = max(0, previous_end - start - 1)
            last = end - start - 1
            n_tokens = last - first
            total_kl += kl_per_token[0, first:last].sum().item()
            total_js += js_per_token[0, first:last].sum().item()
            total_top1_match += top1_match[0, first:last].sum().item()
            total_top5_overlap += top5_overlap[0, first:last].sum().item()
            total_tokens += n_tokens
            previous_end = end
            if end == seq_len:
                break

    if total_tokens == 0:
        return {
            "kl_divergence": float("inf"),
            "js_divergence": float("inf"),
            "top1_agreement": 0.0,
            "top5_overlap": 0.0,
            "total_tokens": 0,
        }

    return {
        "kl_divergence": round(total_kl / total_tokens, 6),
        "js_divergence": round(total_js / total_tokens, 6),
        "top1_agreement": round(total_top1_match / total_tokens, 4),
        "top5_overlap": round(total_top5_overlap / total_tokens, 4),
        "total_tokens": int(total_tokens),
    }


@torch.no_grad()
def token_by_token_fidelity(
    original_model: PreTrainedModel,
    pruned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: Optional[torch.device] = None,
    top_k: int = 10,
) -> List[Dict]:
    """Compare predictions token-by-token for a single text.

    Useful for debugging -- shows exactly where the pruned model diverges.

    Args:
        original_model: Un-pruned model.
        pruned_model: Pruned model.
        tokenizer: Shared tokenizer.
        text: Input text to analyze.
        device: Torch device.
        top_k: How many top predictions to show.

    Returns:
        List of dicts, one per token position, with keys:
        - position, actual_token, original_top1, pruned_top1
        - top1_match, original_probs, pruned_probs_at_original_top
    """
    if device is None:
        device = get_device()

    original_model.eval()
    pruned_model.eval()

    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    seq_len = input_ids.size(1)

    results = []

    for i in range(seq_len - 1):
        context = input_ids[:, :i+1]
        actual_next = input_ids[0, i+1].item()

        orig_out = original_model(context)
        pruned_out = pruned_model(context)

        orig_logits = orig_out.logits[0, -1, :]   # (vocab,)
        pruned_logits = pruned_out.logits[0, -1, :]

        orig_probs = torch.softmax(orig_logits, dim=-1)
        pruned_probs = torch.softmax(pruned_logits, dim=-1)

        _, orig_top_idx = orig_logits.topk(top_k)
        orig_top_vals = orig_probs[orig_top_idx]
        pruned_at_orig_top = pruned_probs[orig_top_idx]

        orig_top1 = orig_logits.argmax().item()
        pruned_top1 = pruned_logits.argmax().item()

        results.append({
            "position": i,
            "actual_token": tokenizer.decode([actual_next]),
            "original_top1": tokenizer.decode([orig_top1]),
            "pruned_top1": tokenizer.decode([pruned_top1]),
            "top1_match": orig_top1 == pruned_top1,
            "original_probs": [
                {"token": tokenizer.decode([idx.item()]), "prob": round(val.item(), 4)}
                for val, idx in zip(orig_top_vals, orig_top_idx)
            ],
            "pruned_probs_at_original_top": [
                round(p.item(), 4) for p in pruned_at_orig_top
            ],
        })

    return results


# ===========================================================================
# Perplexity (for reference)
# ===========================================================================

@torch.no_grad()
def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 256,
    stride: int = 128,
    device: Optional[torch.device] = None,
) -> float:
    """Standard perplexity over evaluation texts (sliding windows).

    Lower = model is less "surprised" by real text.
    """
    if device is None:
        device = get_device()

    if not 0 < stride < max_length:
        raise ValueError("Require 0 < stride < max_length for continuous next-token coverage.")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Evaluating perplexity", unit="text"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = enc.input_ids.to(device)
        seq_len = input_ids.size(1)

        previous_end = 0
        for start in range(0, seq_len, stride):
            end = min(start + max_length, seq_len)
            if end - start < 2:
                break

            window = input_ids[:, start:end]
            labels = window.clone()
            labels[:, :max(1, previous_end - start)] = -100
            outputs = model(window, labels=labels)
            loss = outputs.loss

            if loss is not None:
                n_targets = end - max(previous_end, start + 1)
                total_loss += loss.item() * n_targets
                total_tokens += n_targets
            previous_end = end
            if end == seq_len:
                break

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


@torch.no_grad()
def quick_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str = "The quick brown fox jumps over the lazy dog.",
    device: Optional[torch.device] = None,
) -> float:
    """Single-text perplexity for rapid iteration."""
    return evaluate_perplexity(model, tokenizer, [text], max_length=64, stride=32, device=device)


# ===========================================================================
# Latency benchmarking
# ===========================================================================

@torch.no_grad()
def benchmark_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str = "Once upon a time",
    num_tokens: int = 20,
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Measure inference latency for autoregressive generation."""
    if device is None:
        device = get_device()

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(warmup_runs):
        _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)

    times = []
    generated_counts = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        generated_counts.append(output.shape[-1] - inputs.input_ids.shape[-1])

    mean_s = statistics.mean(times)
    return {
        "mean_latency_s": round(mean_s, 4),
        "median_latency_s": round(statistics.median(times), 4),
        "tokens_per_second": round(sum(generated_counts) / sum(times), 2) if sum(times) > 0 else float("inf"),
        "per_token_ms": round(sum(times) / max(1, sum(generated_counts)) * 1000, 2),
    }


@torch.no_grad()
def benchmark_forward_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str = "Once upon a time",
    num_runs: int = 20,
    warmup_runs: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Measure latency of a single forward pass."""
    if device is None:
        device = get_device()

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(warmup_runs):
        _ = model(**inputs)

    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _ = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_forward_ms": round(statistics.mean(times), 3),
        "median_forward_ms": round(statistics.median(times), 3),
    }


# ===========================================================================
# Comprehensive comparison report
# ===========================================================================

class PruningReport:
    """Compare baseline vs pruned using distribution divergence.

    Computes KL / JS divergence, top-k overlap between the original and
    pruned model's output distributions.  Also reports resource metrics
    and standard perplexity for reference.

    Usage:
        report = PruningReport("Qwen2.5-1.5B", "50% Heads Pruned")
        report.record_baseline(baseline_model, tokenizer)
        # ... apply pruning ...
        report.record_pruned(pruned_model, original_model, tokenizer)
        report.print()
    """

    def __init__(self, baseline_label: str = "Baseline",
                 pruned_label: str = "Pruned") -> None:
        self.baseline_label = baseline_label
        self.pruned_label = pruned_label
        self._baseline: Dict[str, float] = {}
        self._pruned: Dict[str, float] = {}
        self._baseline_model: Optional[PreTrainedModel] = None
        self._eval_texts: List[str] = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subfield of artificial intelligence.",
        ]

    @staticmethod
    def _measure(model: nn.Module, device: torch.device) -> Dict[str, float]:
        return {
            "total_params": count_parameters(model),
            "nonzero_params": count_nonzero_parameters(model),
            "model_size_mb": round(model_size_mb(model), 2),
            "memory_rss_mb": round(get_process_memory_mb(), 2),
        }

    def record_baseline(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = get_device()
        self._baseline_model = model
        self._baseline = self._measure(model, device)
        self._baseline["perplexity"] = round(
            evaluate_perplexity(model, tokenizer, self._eval_texts, device=device), 3
        )
        lat = benchmark_latency(model, tokenizer, device=device)
        self._baseline.update({f"latency_{k}": v for k, v in lat.items()})

    def record_pruned(
        self,
        pruned_model: PreTrainedModel,
        original_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = get_device()
        if original_model is None:
            original_model = self._baseline_model
        if tokenizer is None:
            raise ValueError("tokenizer is required for record_pruned")

        self._pruned = self._measure(pruned_model, device)
        self._pruned["perplexity"] = round(
            evaluate_perplexity(pruned_model, tokenizer, self._eval_texts, device=device), 3
        )

        # -- Distribution divergence --
        if original_model is not None:
            div = compute_distribution_divergence(
                original_model, pruned_model, tokenizer, self._eval_texts, device=device
            )
            self._pruned["kl_divergence_vs_original"] = div["kl_divergence"]
            self._pruned["js_divergence_vs_original"] = div["js_divergence"]
            self._pruned["top1_agreement"] = div["top1_agreement"]
            self._pruned["top5_overlap"] = div["top5_overlap"]

        lat = benchmark_latency(pruned_model, tokenizer, device=device)
        self._pruned.update({f"latency_{k}": v for k, v in lat.items()})

    def print(self) -> None:
        """Print a formatted side-by-side comparison."""
        print("\n" + "=" * 72)
        print(f"  PRUNING REPORT: {self.baseline_label}  vs  {self.pruned_label}")
        print("=" * 72)
        all_keys = sorted(set(list(self._baseline.keys()) + list(self._pruned.keys())))
        print(f"{'Metric':<30} {'Baseline':>18} {'Pruned':>18}")
        print("-" * 66)
        for key in all_keys:
            b = self._baseline.get(key, "N/A")
            p = self._pruned.get(key, "N/A")

            def fmt(v):
                if isinstance(v, float):
                    return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
                if isinstance(v, int) and v > 1000:
                    return f"{v:,}"
                return str(v)

            print(f"{key:<30} {fmt(b):>18} {fmt(p):>18}")
        print("=" * 72 + "\n")
