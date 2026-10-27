"""
End-to-end experiment running from YAML configuration files.

Loads two model copies (baseline + pruned) so distribution divergence
is properly measured against the untouched original.

Usage:
    python examples/run_experiment.py config/pruning_configs/head_pruning.yaml
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config_loader import load_yaml, create_pruner_from_config
from src.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer
from src.evaluation import PruningReport
from src.utils import count_parameters, model_size_mb, get_process_memory_mb

# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python run_experiment.py <path/to/config.yaml>")
    print("\nAvailable configs:")
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config", "pruning_configs")
    for f in sorted(os.listdir(config_dir)):
        if f.endswith(".yaml"):
            print(f"  config/pruning_configs/{f}")
    sys.exit(1)

config_path = sys.argv[1]

# ---------------------------------------------------------------------------
config = load_yaml(config_path)
model_name = config.get("model_name", DEFAULT_MODEL_NAME)

print(f"\n{'='*60}")
print(f"  EXPERIMENT: {os.path.basename(config_path)}")
print(f"  Type: {config.get('type')}  |  Mode: {config.get('mode')}")
print(f"{'='*60}\n")

# Load TWO copies: baseline stays untouched, pruned gets modified
print("Loading baseline model (untouched) ...")
baseline_model, tokenizer = load_model_and_tokenizer(model_name)
print("Loading model to prune (fresh copy) ...")
pruned_model, _ = load_model_and_tokenizer(model_name)

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
report = PruningReport("Original", "Pruned")
report.record_baseline(baseline_model, tokenizer)

print(f"\nBaseline — Params: {count_parameters(baseline_model):,}  |  "
      f"Size: {model_size_mb(baseline_model):.2f} MB")
print(f"  Memory: {get_process_memory_mb():.2f} MB")

# ---------------------------------------------------------------------------
# Apply pruning to the pruned copy only
# ---------------------------------------------------------------------------
pruner = create_pruner_from_config(pruned_model, config)
pruner.snapshot()

mode = config.get("mode", "masking")
if mode == "masking":
    pruner.apply_masking()
else:
    pruner.apply_structural()

# ---------------------------------------------------------------------------
# Evaluate: compare pruned vs original
# ---------------------------------------------------------------------------
report.record_pruned(pruned_model, original_model=baseline_model, tokenizer=tokenizer)
report.print()

pruner.revert()
print("[OK] Model reverted. Experiment complete.")
