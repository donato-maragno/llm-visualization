# Historical documentation — pre-coding/judge workflow (not the current app)

## Local web interface

From this folder, use the **existing workspace environment**:

```powershell
& ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
& ..\.venv\Scripts\python.exe run_web.py
```

Open **http://127.0.0.1:8000**. On Windows you can also double-click
`start_web.bat`. Use `run_web.py --port 8001` for a different port.
There is no Node/npm build, CDN, or external font dependency.

The app provides:

- Ten editable questions per topic; deselect, remove, add your own, or generate
  a fresh question locally. Generated questions are reviewed before running.
- Up to eight variants: last-layer bypass, per-layer head masking, magnitude
  MLP pruning, and random MLP pruning, each with an adjustable strength.
- Live progress and a question-by-variant score matrix, aggregate scores,
  original/pruned answers side by side, timings, token-cap warnings, and
  the judge's explanation for every question and variant.
- A free **local self-judge**, optional **OpenRouter judge**, or **answers only**.
  The local 1.5B judge is not independent or calibrated; repetitive scores
  are flagged. Similarity is not factual correctness or a quality guarantee.
- Safe cancellation, saved history, and JSON export. Partial results are
  saved after each answer/judgment in `results/web_runs/`. Refreshing the
  browser reconnects to an active task. History survives server restarts;
  interrupted tasks are marked as such, not silently resumed.

### GPT-2-only live model view

Select **GPT-2 · 124M · live model view** under Model. Qwen remains the
default, recommended model; its architecture is deliberately not visualized.
The GPT-2 panel displays all twelve transformer layers and twelve heads per
layer. During generation it highlights the actual applied layer bypasses,
head-output masks, or MLP weight masks. Click a layer for exact tensor names,
shapes and masked-position counts, plus the first 96 mask positions from each
tensor. These tiles are a sample, not a picture of the entire weight matrix.

Live view follows the server's current phase (baseline, applying, pruned
generation, restoration, judging). Token progress is updated as generation
proceeds, with browser polling roughly once per second. **Follow live pruning**
shows the restored model after pruning ends. **Inspect saved mask** switches
to a clearly labeled recorded snapshot, also available in saved runs.

This is topology/mask visualization, **not** a neural-activation trace or
physical parameter removal. All components remain dense. Baseline answers
are generated first; comparison cells show “Baseline first”, “Queued”,
“Generating”, or “Awaiting judge” instead of an ambiguous “Pending”.

GPT-2 is not instruction tuned and will produce much weaker answers than
Qwen. For GPT-2 runs the local judge is **unpruned Qwen**, not GPT-2; the
service switches models sequentially rather than holding two GPU copies.
New-question generation also uses Qwen. Choose “Answers only” for the
fastest architecture exploration without judge/model-switch overhead.

Only **one worker/process** may own the model. The model loads lazily on the
first task. GPU tasks are serialized; the unpruned model is restored before
local judging and between variants, including on errors and cancellation.
Greedy decoding reduces randomness in comparisons. Random pruning uses the
selected seed. Head selection is the first N heads, not importance-ranked.
Weight percentages apply to MLPs only, with minor FP16 threshold ties.
Layer bypass preserves the residual stream but still computes the original
block: masking is an experiment, not a speed/memory optimization.

The server binds to **127.0.0.1 only** and validates hosts and mutation tokens.
It is not a multi-user/public production service. Do not expose it through
a public proxy. Run configuration and generated answers are stored locally;
API keys are not included in result files or sent to the browser.

### Optional external judge

Set credentials **before starting the server**:

```powershell
$env:OPENROUTER_API_KEY = "your-key"
$env:JUDGE_MODEL = "your-provider/model-id"
& ..\.venv\Scripts\python.exe run_web.py
```

Choose OpenRouter in the UI, verify a currently available model ID, and
confirm the explicit data-sharing/cost notice. Each question/variant pair
makes one judge request. There are **no OpenRouter calls by default**.
API or malformed-response errors are shown separately and excluded from
averages; they never become a neutral `0` score. Cancellation of an in-flight
external request may wait for its response or the 120-second timeout.

**Security:** a previously embedded API credential was removed during the
review. Revoke/rotate that old key; removing it from source cannot undo
earlier disclosure.

### Tests and review

```powershell
& ..\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& ..\.venv\Scripts\python.exe -m pytest tests/ -q
```

Unit tests use small in-memory models and fake judge responses, with no
downloads or API charges. `tests/browser_smoke.py` is an optional real-browser
test requiring Playwright and Microsoft Edge; it runs actual local inference
against a running server. See `CODE_REVIEW.md` for findings, fixes, and the
remaining limitations of the older CLI/structural helpers.

---

A modular framework for experimenting with neural-network pruning on
open-weight LLMs, and measuring how pruning affects **output meaning**
using an external judge LLM.

**Quick start:** load the free, open-weight Qwen2.5-1.5B-Instruct model, prune half its attention heads, and
get a semantic-similarity score (from -1 to 1) comparing the pruned
model's answers against the unpruned model's.


## How output quality is measured

Instead of perplexity or logit divergence, we ask a **larger judge LLM**
(GPT-6 Luna via OpenRouter) to compare answers:

1. Ask the same **10 questions** (parameterized by a `topic`) to both
   the unpruned and pruned model.
2. Send each pair of answers to the judge.
3. The judge returns a **similarity score from -1 to 1**:

| Score | Meaning |
|-------|---------|
| `1`   | The two answers mean the **same** thing |
| `0`   | The two answers mean **different / unrelated** things |
| `-1`  | The two answers mean **opposite** things |

The average across all 10 questions is the pruning-quality score.


## Project Structure

```
pruning_experiments/
├── run_all.py                  # Run all strategies, compare judge scores
├── requirements.txt
├── config/
│   ├── base_config.yaml
│   ├── config_loader.py
│   └── pruning_configs/        # YAML pruning configs
├── src/
│   ├── model_loader.py         # Load Qwen2.5 and compatible causal LMs
│   ├── judge_eval.py           # Judge-LLM semantic similarity (NEW)
│   ├── utils.py
│   ├── evaluation.py           # (legacy) perplexity / divergence
│   ├── masking.py              # Logical masking helpers
│   ├── structural.py           # Structural pruning helpers
│   └── pruners/
│       ├── base_pruner.py
│       ├── module_pruner.py    # Layer pruning
│       ├── head_pruner.py      # Attention-head pruning
│       └── weight_pruner.py    # Weight pruning
├── examples/
│   ├── judge_experiment.py     # Judge-based evaluation (NEW)
│   └── ...
└── tests/
```


## Installation

```bash
cd pruning_experiments
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux/macOS

pip install -r requirements.txt
```

For CLI judge experiments, provide your own OpenRouter key; there is no
embedded/default key:

```bash
set OPENROUTER_API_KEY=sk-or-v1-...
```


## Models

| Model | Params | Layers | Heads | Notes |
|-------|--------|--------|-------|-------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | 28 | 12 (GQA: 2 KV) | **Recommended; FP16 fits an 8 GB RTX 4070** |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | 24 | 14 (2 KV) | Faster, weaker |
| `Qwen/Qwen2.5-3B-Instruct` | 3.09B | 36 | 16 (2 KV) | Stronger, but tighter VRAM headroom |

The default Qwen2.5-1.5B-Instruct is Apache 2.0 licensed and can be run locally without
paying for model inference. The initial download requires an internet
connection. The optional semantic judge described below is a separate
OpenRouter API service and is not part of local model inference.

> **GQA note:** Qwen2.5 uses Grouped Query Attention (12 Q heads share
> 2 KV heads). Pruning a Q head zeroes only that head's slice; pruning a
> KV head affects 6 Q heads at once.


## Quick Start

```bash
# Compare pruning techniques locally (no OpenRouter/API call)
python examples/local_pruning_benchmark.py

# Run all pruning strategies and compare judge scores
python run_all.py --topic "climate change"

# Or just one strategy (10 questions)
python examples/judge_experiment.py --topic "physics"
```

Each run:
1. Generates 10 questions about the topic.
2. Gets baseline answers from the unpruned model.
3. Applies each pruning strategy, gets pruned answers, judges similarity.
4. Prints a summary table of average similarity per strategy.

The local benchmark compares KL/JS distribution divergence, next-token
agreement, and perplexity for layer, attention-head, and MLP-weight pruning
at several strengths. Its short built-in text set is intended for a quick
comparison, not as a general language-model benchmark.


## Pruning Mechanisms

### Module / Layer pruning
```python
ModulePruner(model, {
    "target_modules": ["model.layers.26", "model.layers.27"],
    "mode": "masking",       # or "structural"
})
```

### Attention-head pruning
```python
HeadPruner(model, {
    "head_indices": "0:6",   # heads 0-5 in all layers
    "mode": "masking",
})
# Or per-layer: {0: [0,1], 5: "even", 10: "0:6"}
```

### Weight pruning
```python
WeightPruner(model, {
    "strategy": "magnitude",  # magnitude | random | threshold | custom
    "sparsity": 0.5,
    "target_modules": "all",  # or ["model.layers.*.mlp.*"]
    "mode": "masking",
})
```


## Masking vs Structural

| Aspect | Logical Masking | Structural Pruning |
|--------|----------------|-------------------|
| Model graph | Unchanged | Modified |
| Param count | Same | Lower |
| Inference speed | No gain | Real speed-up |
| Reversible | Cheap (remove hooks) | Needs backup |
| Use case | Exploration | Deployment |


## Judge API (OpenRouter)

The judge call is isolated in `src/judge_eval.py::call_judge()`. It uses:

- Model: `openai/gpt-6-luna` (a reasoning model — needs ~2000+ tokens
  for reasoning before it emits the JSON score).
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`

Configure the CLI judge with the `JUDGE_MODEL` and `OPENROUTER_API_KEY`
environment variables. Verify model availability with your provider; the
web UI also allows changing the OpenRouter model ID per run.
