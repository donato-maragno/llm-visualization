"""Run no-API local pruning comparisons against the unpruned model.

Usage::

    python examples/local_pruning_benchmark.py

The benchmark reports distribution divergence, next-token agreement, and
perplexity for several layer-, head-, and MLP-weight-pruning strengths.
It uses one model copy and does not call the external OpenRouter judge.
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model_loader import DEFAULT_MODEL_NAME, get_model_metadata, load_model_and_tokenizer
from src.pruners.head_pruner import HeadPruner
from src.pruners.module_pruner import ModulePruner
from src.pruners.weight_pruner import WeightPruner


EVAL_TEXTS = [
    "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce sugars and oxygen.",
    "A planetary atmosphere is shaped by gravity, temperature, chemistry, and interactions with radiation from its star.",
    "In machine learning, a model is trained by adjusting its parameters to reduce errors on examples from a dataset.",
]
MAX_TOKENS_PER_TEXT = 128


def prepare_batches(tokenizer):
    batches = []
    for text in EVAL_TEXTS:
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS_PER_TEXT,
        ).input_ids
        if input_ids.shape[1] > 1:
            batches.append(input_ids)
    if not batches:
        raise RuntimeError("The evaluation texts produced no next-token targets.")
    return batches


@torch.inference_mode()
def collect_reference(model, batches, device):
    """Cache unpruned logits on CPU and calculate reference perplexity."""
    logits = []
    total_nll = 0.0
    total_tokens = 0
    for input_ids in batches:
        input_ids = input_ids.to(device)
        outputs = model(input_ids, labels=input_ids)
        logits.append(outputs.logits[0, :-1].detach().to(device="cpu", dtype=torch.float32))
        n_tokens = input_ids.shape[1] - 1
        total_nll += outputs.loss.float().item() * n_tokens
        total_tokens += n_tokens
        del outputs
    return logits, torch.exp(torch.tensor(total_nll / total_tokens)).item()


@torch.inference_mode()
def score_model(model, batches, reference_logits, device):
    total_kl = 0.0
    total_js = 0.0
    total_top1 = 0
    total_tokens = 0
    total_nll = 0.0

    for input_ids, baseline_logits in zip(batches, reference_logits):
        input_ids = input_ids.to(device)
        outputs = model(input_ids, labels=input_ids)
        current_logits = outputs.logits[0, :-1].detach().to(device="cpu", dtype=torch.float32)

        baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        baseline_probs = baseline_log_probs.exp()
        current_probs = current_log_probs.exp()
        mixture = (baseline_probs + current_probs) * 0.5
        log_mixture = mixture.clamp_min(1e-30).log()

        kl = (baseline_probs * (baseline_log_probs - current_log_probs)).sum(dim=-1)
        js = 0.5 * (
            baseline_probs * (baseline_log_probs - log_mixture)
        ).sum(dim=-1) + 0.5 * (
            current_probs * (current_log_probs - log_mixture)
        ).sum(dim=-1)
        total_kl += kl.sum().item()
        total_js += js.sum().item()
        total_top1 += (
            baseline_logits.argmax(dim=-1) == current_logits.argmax(dim=-1)
        ).sum().item()

        n_tokens = input_ids.shape[1] - 1
        total_nll += outputs.loss.float().item() * n_tokens
        total_tokens += n_tokens
        del outputs

    return {
        "kl": total_kl / total_tokens,
        "js": total_js / total_tokens,
        "top1": total_top1 / total_tokens,
        "ppl": torch.exp(torch.tensor(total_nll / total_tokens)).item(),
        "tokens": total_tokens,
    }


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL_NAME)
    metadata = get_model_metadata(model)
    device = next(model.parameters()).device
    batches = prepare_batches(tokenizer)
    reference_logits, baseline_ppl = collect_reference(model, batches, device)

    print(f"\nModel: {DEFAULT_MODEL_NAME}")
    print(f"Device: {device}; architecture: {metadata}")
    print(f"Baseline perplexity: {baseline_ppl:.4f}")

    layer_count = metadata["num_layers"]
    head_count = metadata["num_heads"]
    experiments = []
    for num_layers in (1, 2, 4):
        targets = [f"model.layers.{i}" for i in range(layer_count - num_layers, layer_count)]
        experiments.append((
            f"Layer bypass: last {num_layers} layer(s)",
            ModulePruner(model, {
                "target_modules": targets, "mode": "masking",
                "masking_replacement": "identity",
            }),
        ))
    for fraction in (0.25, 0.50):
        num_heads = max(1, int(head_count * fraction))
        experiments.append((
            f"Head mask: {num_heads}/{head_count} heads per layer",
            HeadPruner(model, {
                "head_indices": list(range(num_heads)),
                "mode": "masking",
            }),
        ))
    for strategy in ("magnitude", "random"):
        for sparsity in (0.20, 0.50, 0.80):
            experiments.append((
                f"MLP {strategy}: {sparsity:.0%} sparsity",
                WeightPruner(model, {
                    "strategy": strategy,
                    "sparsity": sparsity,
                    "target_modules": ["model.layers.*.mlp.*"],
                    "mode": "masking",
                }),
            ))

    print("\nPruning result vs. unpruned model (lower KL/JS/PPL is better; higher top1 is better)")
    print(f"{'Technique / strength':<42} {'KL':>10} {'JS':>10} {'Top-1':>9} {'PPL':>10}")
    print("-" * 85)
    print(f"{'Unpruned baseline':<42} {0.0:>10.5f} {0.0:>10.5f} {1.0:>9.1%} {baseline_ppl:>10.4f}")

    results = []
    for label, pruner in experiments:
        try:
            pruner.apply_masking()
            scores = score_model(model, batches, reference_logits, device)
        finally:
            pruner.revert()
        print(
            f"{label:<42} {scores['kl']:>10.5f} {scores['js']:>10.5f} "
            f"{scores['top1']:>9.1%} {scores['ppl']:>10.4f}"
        )
        results.append({"variant": label, **scores})

    print(f"\nScored {scores['tokens']} next-token positions per variant.")
    print("These are local distribution/perplexity measurements, not human-judged answer quality.")
    restored = score_model(model, batches, reference_logits, device)
    if restored["kl"] > 1e-6 or restored["top1"] != 1.0:
        raise RuntimeError(f"Model restoration check failed: {restored}")
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "local_pruning_benchmark.json"
    output.write_text(json.dumps({
        "model": DEFAULT_MODEL_NAME,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__,
        "texts": EVAL_TEXTS,
        "baseline_perplexity": baseline_ppl,
        "results": results,
        "restoration_check": restored,
    }, indent=2), encoding="utf-8")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
