"""
Module / Layer pruning example.

Demonstrates both logical masking and structural pruning of entire
transformer layers.  Loads two separate copies of the model so that
distribution divergence can be computed against the untouched original.

Usage:
    python examples/module_pruning_example.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_loader import DEFAULT_MODEL_NAME, get_model_metadata, load_model_and_tokenizer
from src.pruners.module_pruner import ModulePruner
from src.evaluation import PruningReport, compute_distribution_divergence
from src.utils import count_parameters, get_device

MODEL_NAME = DEFAULT_MODEL_NAME

# ---------------------------------------------------------------------------
print("=" * 60)
print("  MODULE PRUNING EXAMPLE")
print("=" * 60)

# Load TWO separate copies: one untouched baseline, one to prune.
# This is needed so we can compare the pruned model's output distribution
# against the original's — not just against ground-truth text.
print("Loading baseline model (untouched) ...")
baseline_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
print("Loading model to prune (fresh copy) ...")
pruned_model, _ = load_model_and_tokenizer(MODEL_NAME)
metadata = get_model_metadata(pruned_model)

# ---------------------------------------------------------------------------
# Baseline metrics
# ---------------------------------------------------------------------------
report = PruningReport("Original Qwen2.5-1.5B", "Layer-Pruned Qwen2.5-1.5B")
report.record_baseline(baseline_model, tokenizer)

print("\n--- Before Pruning ---")
print(f"Parameters: {count_parameters(pruned_model):,}")

# ---------------------------------------------------------------------------
# 1) LOGICAL MASKING — zero the output of the last 2 layers
# ---------------------------------------------------------------------------
print("\n--- Experiment 1: Logical masking (last 2 layers) ---")
pruner = ModulePruner(pruned_model, {
    "target_modules": [
        f"model.layers.{metadata['num_layers'] - 2}",
        f"model.layers.{metadata['num_layers'] - 1}",
    ],
    "mode": "masking",
})
pruner.apply_masking()

# Compare pruned model's output distribution to the untouched original
div = compute_distribution_divergence(
    baseline_model, pruned_model, tokenizer,
    ["The quick brown fox jumps over the lazy dog."],
    device=get_device(),
)
print(f"KL divergence (masked vs original): {div['kl_divergence']}")
print(f"Top-1 agreement:                   {div['top1_agreement']:.1%}")
print(f"Top-5 overlap:                     {div['top5_overlap']:.1%}")

pruner.revert()
print("Reverted.\n")

# ---------------------------------------------------------------------------
# 2) STRUCTURAL PRUNING — replace the final layer with Identity
# ---------------------------------------------------------------------------
print("--- Experiment 2: Structural pruning (last layer -> Identity) ---")
pruner2 = ModulePruner(pruned_model, {
    "target_modules": [f"model.layers.{metadata['num_layers'] - 1}"],
    "mode": "structural",
    "structural_replacement": "identity",
})
pruner2.apply_structural()

# Full report: resource metrics + distribution divergence vs original
report.record_pruned(pruned_model, original_model=baseline_model, tokenizer=tokenizer)
report.print()

pruner2.revert()
print("[OK] Module pruning example complete.")
