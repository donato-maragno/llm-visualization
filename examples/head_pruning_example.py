"""
Attention-head pruning example.

Demonstrates pruning individual attention heads using both logical
masking and structural zeroing.  Loads two model copies so distribution
divergence is computed against the untouched original.

Usage:
    python examples/head_pruning_example.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_loader import DEFAULT_MODEL_NAME, get_model_metadata, load_model_and_tokenizer
from src.pruners.head_pruner import HeadPruner
from src.evaluation import PruningReport, compute_distribution_divergence
from src.utils import get_device

MODEL_NAME = DEFAULT_MODEL_NAME

# ---------------------------------------------------------------------------
print("=" * 60)
print("  ATTENTION HEAD PRUNING EXAMPLE")
print("=" * 60)

# Load two copies: one stays untouched, one gets pruned
print("Loading baseline model ...")
baseline_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
print("Loading model to prune ...")
pruned_model, _ = load_model_and_tokenizer(MODEL_NAME)
metadata = get_model_metadata(pruned_model)

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
report = PruningReport("Original Qwen2.5-1.5B", "Head-Pruned Qwen2.5-1.5B")
report.record_baseline(baseline_model, tokenizer)

eval_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subfield of artificial intelligence.",
]

# ---------------------------------------------------------------------------
# Experiment 1: MASK heads 0-5 in every layer (50% of heads)
# ---------------------------------------------------------------------------
print(f"\n--- Experiment 1: Mask half the heads in all {metadata['num_layers']} layers ---")

pruner = HeadPruner(pruned_model, {
    "head_indices": f"0:{metadata['num_heads'] // 2}",
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
# Experiment 2: STRUCTURALLY zero heads 10,11 in layers 0-5
# ---------------------------------------------------------------------------
print("--- Experiment 2: Structurally zero the last two heads in the first six layers ---")

per_layer = {
    i: [metadata["num_heads"] - 2, metadata["num_heads"] - 1]
    for i in range(min(6, metadata["num_layers"]))
}
pruner2 = HeadPruner(pruned_model, {
    "head_indices": per_layer,
    "mode": "structural",
    "zero_output_projection": True,
})
pruner2.snapshot()
pruner2.apply_structural()

report.record_pruned(pruned_model, original_model=baseline_model, tokenizer=tokenizer)
report.print()

pruner2.revert()

# ---------------------------------------------------------------------------
# Experiment 3: Targeted masking — even heads in layer 0 only
# ---------------------------------------------------------------------------
print("--- Experiment 3: Mask even heads in layer 0 only ---")

pruner3 = HeadPruner(pruned_model, {
    "head_indices": {0: "even"},
    "mode": "masking",
})
pruner3.apply_masking()

div3 = compute_distribution_divergence(
    baseline_model, pruned_model, tokenizer,
    ["The quick brown fox jumps over the lazy dog."],
    device=get_device(),
)
print(f"KL divergence:   {div3['kl_divergence']}")
print(f"Top-1 agreement: {div3['top1_agreement']:.1%}")

pruner3.revert()
print("\n[OK] Head pruning example complete.")
