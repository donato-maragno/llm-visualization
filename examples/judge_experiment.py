"""
Semantic similarity evaluation using an external judge LLM via OpenRouter.

Memory-efficient workflow:
1. Load ONE Qwen2.5 model.
2. Generate baseline answers for 10 questions on a topic.
3. Apply a pruning strategy (masking), generate pruned answers, and
   have GPT-6 Luna judge semantic similarity against the baseline.
4. Compare multiple strategies side by side.

Usage:
    python examples/judge_experiment.py [--topic "climate change"]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer, get_model_metadata
from src.pruners.head_pruner import HeadPruner
from src.pruners.module_pruner import ModulePruner
from src.pruners.weight_pruner import WeightPruner
from src.judge_eval import (
    generate_questions,
    generate_baseline_answers,
    judge_against_baseline,
)

# ---------------------------------------------------------------------------
TOPIC = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--topic" else "artificial intelligence"
NUM_QUESTIONS = 10
MODEL_NAME = DEFAULT_MODEL_NAME

print("=" * 60)
print("  JUDGE-BASED PRUNING EVALUATION")
print(f"  Model: {MODEL_NAME}")
print(f"  Topic: {TOPIC}")
print(f"  Questions per strategy: {NUM_QUESTIONS}")
print(f"  Judge: openai/gpt-6-luna (score -1 to 1)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Load model + baseline answers (once)
# ---------------------------------------------------------------------------
print("\nLoading model ...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
metadata = get_model_metadata(model)
print("Architecture:", metadata)

questions = generate_questions(TOPIC)[:NUM_QUESTIONS]
print("\nGenerating baseline answers (unpruned) ...")
baseline_answers = generate_baseline_answers(model, tokenizer, questions)

# ---------------------------------------------------------------------------
# Run each strategy against the stored baseline
# ---------------------------------------------------------------------------
strategies = {
    "0. No pruning (sanity)": None,
    "1. Mask last 2 layers": ModulePruner(model, {
        "target_modules": [
            f"model.layers.{metadata['num_layers'] - 2}",
            f"model.layers.{metadata['num_layers'] - 1}",
        ],
        "mode": "masking",
    }),
    "2. Mask 50% of attention heads": HeadPruner(model, {
        "head_indices": f"0:{metadata['num_heads'] // 2}",
        "mode": "masking",
    }),
    "3. 50% magnitude MLP weight pruning": WeightPruner(model, {
        "strategy": "magnitude",
        "sparsity": 0.5,
        "target_modules": ["model.layers.*.mlp.*"],
        "mode": "masking",
    }),
}

results = {}

for name, pruner in strategies.items():
    print("\n" + "-" * 55)
    print(f"STRATEGY: {name}")
    print("-" * 55)

    if pruner is None:
        result = judge_against_baseline(model, tokenizer, questions, baseline_answers)
    else:
        pruner.apply_masking()
        result = judge_against_baseline(model, tokenizer, questions, baseline_answers)
        pruner.revert()

    results[name] = result
    print(f"\n  >>> {name}: average similarity = {result['average_score']:+.3f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print(f"  Topic: {TOPIC}  |  Questions: {NUM_QUESTIONS}")
print("=" * 60)
for name, r in results.items():
    print(f"  {name:<40} {r['average_score']:>+7.3f}")
print("=" * 60)
