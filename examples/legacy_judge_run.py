"""
Judge-based evaluation of pruning strategies using 10 questions.

Memory-efficient workflow:
1. Load ONE model.
2. Generate baseline answers for 10 questions (stored as text).
3. For each strategy, apply masking, generate pruned answers,
   judge against the baseline, then revert.

Usage:
    python examples/legacy_judge_run.py [--topic "climate change"]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_loader import load_model_and_tokenizer, get_model_metadata
from src.pruners.head_pruner import HeadPruner
from src.pruners.module_pruner import ModulePruner
from src.pruners.weight_pruner import WeightPruner
from src.judge_eval import (
    generate_questions,
    generate_baseline_answers,
    judge_against_baseline,
)

TOPIC = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--topic" else "artificial intelligence"
NUM_QUESTIONS = 10
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 60)
print("  JUDGE-BASED PRUNING EVALUATION")
print(f"  Model: {MODEL_NAME}")
print(f"  Topic: {TOPIC}")
print(f"  Questions: {NUM_QUESTIONS}")
print("=" * 60)

# -------------------------------------------------------------------
# 1. Load model + generate baseline answers (once)
# -------------------------------------------------------------------
print("\nLoading model ...")
model, tok = load_model_and_tokenizer(MODEL_NAME)
print("Architecture:", get_model_metadata(model))

questions = generate_questions(TOPIC)[:NUM_QUESTIONS]

print("\nGenerating baseline answers (unpruned model) ...")
baseline_answers = generate_baseline_answers(model, tok, questions)

# -------------------------------------------------------------------
# 2. Evaluate each pruning strategy
# -------------------------------------------------------------------
strategies = {
    "0. No pruning (sanity)": None,
    "1. Mask 2 attention heads (17%)": HeadPruner(model, {
        "head_indices": [0, 1],
        "mode": "masking",
    }),
    "2. Mask 6 attention heads (50%)": HeadPruner(model, {
        "head_indices": "0:6",
        "mode": "masking",
    }),
    "3. Mask last layer": ModulePruner(model, {
        "target_modules": ["model.layers.27"],
        "mode": "masking",
    }),
    "4. 50% magnitude weight (MLP)": WeightPruner(model, {
        "strategy": "magnitude",
        "sparsity": 0.5,
        "target_modules": "model.layers.*.mlp.*",
        "mode": "masking",
    }),
}

results = {}

for name, pruner in strategies.items():
    print("\n" + "-" * 55)
    print(f"STRATEGY: {name}")
    print("-" * 55)

    if pruner is None:
        result = judge_against_baseline(model, tok, questions, baseline_answers)
    else:
        pruner.apply_masking()
        result = judge_against_baseline(model, tok, questions, baseline_answers)
        pruner.revert()

    results[name] = result
    print(f"\n  >>> {name}: average similarity = {result['average_score']:+.3f}")

# -------------------------------------------------------------------
# 3. Final summary
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print(f"  Topic: {TOPIC}  |  Questions: {NUM_QUESTIONS}")
print("=" * 60)
for name, r in results.items():
    print(f"  {name:<40} {r['average_score']:>+7.3f}")
print("=" * 60)
