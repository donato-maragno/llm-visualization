"""
Weight-level pruning example.

Demonstrates magnitude-based, random, and custom weight pruning.
Loads two model copies so distribution divergence is measured against
the untouched original.

Usage:
    python examples/weight_pruning_example.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer
from src.pruners.weight_pruner import WeightPruner
from src.evaluation import PruningReport, compute_distribution_divergence
from src.utils import count_nonzero_parameters, get_device

MODEL_NAME = DEFAULT_MODEL_NAME

# ---------------------------------------------------------------------------
print("=" * 60)
print("  WEIGHT PRUNING EXAMPLE")
print("=" * 60)

print("Loading baseline model ...")
baseline_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
print("Loading model to prune ...")
pruned_model, _ = load_model_and_tokenizer(MODEL_NAME)

eval_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subfield of artificial intelligence.",
]

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
report = PruningReport("Original Qwen2.5-1.5B", "Weight-Pruned Qwen2.5-1.5B")
report.record_baseline(baseline_model, tokenizer)

# ---------------------------------------------------------------------------
# Experiment 1: Magnitude-based 50% sparsity on MLP weights (masking)
# ---------------------------------------------------------------------------
print("\n--- Experiment 1: 50% magnitude pruning of MLP weights (masking) ---")

pruner = WeightPruner(pruned_model, {
    "strategy": "magnitude",
    "sparsity": 0.5,
    "target_modules": ["model.layers.*.mlp.*"],
    "mode": "masking",
})
pruner.apply_masking()

div = compute_distribution_divergence(
    baseline_model, pruned_model, tokenizer, eval_texts, device=get_device(),
)
print(f"KL divergence:   {div['kl_divergence']}")
print(f"JS divergence:   {div['js_divergence']}")
print(f"Top-1 agreement: {div['top1_agreement']:.1%}")
print(f"Top-5 overlap:   {div['top5_overlap']:.1%}")

pruner.revert()
print("Reverted.\n")

# ---------------------------------------------------------------------------
# Experiment 2: 30% random pruning of MLP weights (structural)
# ---------------------------------------------------------------------------
print("--- Experiment 2: 30% random pruning of MLP weights (structural) ---")

pruner2 = WeightPruner(pruned_model, {
    "strategy": "random",
    "sparsity": 0.3,
    "target_modules": ["model.layers.*.mlp.*"],
    "mode": "structural",
})
pruner2.snapshot()
pruner2.apply_structural()

report.record_pruned(pruned_model, original_model=baseline_model, tokenizer=tokenizer)
report.print()

# Per-layer sparsity breakdown
sparsity_report = pruner2.get_sparsity_report()
mlp_layers = {k: v for k, v in sparsity_report.items() if "mlp" in k}
print("MLP per-layer sparsity (~30% target):")
for name in sorted(mlp_layers)[:4]:
    print(f"  {name}: {mlp_layers[name]:.1%}")
print("  ...")

pruner2.revert()

# ---------------------------------------------------------------------------
# Experiment 3: Custom mask — keep every 10th weight in one layer
# ---------------------------------------------------------------------------
print("\n--- Experiment 3: Custom mask (keep every 10th weight) ---")

def every_10th_mask(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    mask = torch.zeros_like(param.data)
    flat = mask.view(-1)
    flat[::10] = 1.0
    return mask

pruner3 = WeightPruner(pruned_model, {
    "strategy": "custom",
    "custom_mask_fn": every_10th_mask,
    "target_modules": "model.layers.0.mlp.up_proj",
    "mode": "structural",
})
pruner3.snapshot()
pruner3.apply_structural()

nonzero = count_nonzero_parameters(pruned_model)
print(f"Non-zero params after custom mask: {nonzero:,}")

pruner3.revert()
print("\n[OK] Weight pruning example complete.")
