"""
Baseline evaluation – load the original model and record metrics.

Run this before any pruning to establish a baseline for comparison.

Usage:
    python examples/baseline_eval.py
"""

import sys
import os

# Allow running examples from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer, get_model_metadata
from src.evaluation import PruningReport, evaluate_perplexity, benchmark_latency
from src.utils import count_parameters, model_size_mb, print_model_structure, get_process_memory_mb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = DEFAULT_MODEL_NAME

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("=" * 60)
print("  BASELINE EVALUATION")
print("=" * 60)

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

# Show architecture overview
meta = get_model_metadata(model)
print("Model metadata:", meta)
print_model_structure(model, max_depth=2)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
print("\n--- Resource Metrics ---")
print(f"Total parameters:     {count_parameters(model):,}")
print(f"Model size (MB):      {model_size_mb(model):.2f}")
print(f"Process memory (MB):  {get_process_memory_mb():.2f}")

# ---------------------------------------------------------------------------
# Output quality (quick perplexity)
# ---------------------------------------------------------------------------
print("\n--- Quality (Perplexity) ---")
eval_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subfield of artificial intelligence that focuses on building systems that learn from data.",
    "Once upon a time in a land far away, there lived a wise old wizard who guarded the secrets of the universe.",
]
ppl = evaluate_perplexity(model, tokenizer, eval_texts)
print(f"Perplexity (3 texts): {ppl:.3f}")

# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
print("\n--- Latency ---")
latency = benchmark_latency(model, tokenizer, num_tokens=30, num_runs=10)
for k, v in latency.items():
    print(f"  {k}: {v}")

print("\n[OK] Baseline evaluation complete.")
