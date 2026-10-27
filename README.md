# LLM visualization

Two separate menu pages: a **fixed GPT-2 sentence walkthrough** for learning how
transformers and pruning work, and a **Qwen-only experiment lab** for comparing
original/pruned Python solutions on real tested questions.
**No judge, API key, or paid inference.**

## Run the web interface (Windows)

### Requirements

- Python installed on your computer, with the `py` or `python` command available.
- An internet connection to install dependencies and download the model on first use.
- A CUDA-capable GPU is optional; the app can use the CPU, but generation is slower.

No Node.js, npm, Docker, or API key is required to run the included interface.
Precompiled visualization assets are checked in. Node/npm are only needed if
you modify and rebuild the frontend source in `frontend/`.

### 1. First-time setup

Open PowerShell in the cloned project folder (`llm-visualization`). The commands below
use a virtual environment named `.venv` in the **parent folder**, matching the
included launcher.

```powershell
# Create the environment only if it does not already exist.
if (-not (Test-Path ..\.venv\Scripts\python.exe)) {
    py -m venv ..\.venv
}

# Install the project's dependencies.
& ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is unavailable but `python` works, use `python -m venv ..\.venv` to
create the environment instead. No environment activation is needed when using
the explicit Python path shown here.

### 2. Start the server

Double-click **`start_web.bat`**, or run this from the project folder in PowerShell:

```powershell
& ..\.venv\Scripts\python.exe run_web.py
```

### 3. Open the app

Open **http://127.0.0.1:8000** in your browser. Keep the server's terminal window
open while using the app; press **Ctrl+C** in that window to stop it.

Use the menu tabs:

- **GPT-2 walkthrough:** http://127.0.0.1:8000/, with a fixed sentence and preset masks.
- **Qwen experiments:** http://127.0.0.1:8000/experiments, with editable coding problems,
  pruning settings, generated solutions and actual test comparisons.

Below the Learn visualization is a chapter-linked guide to Transformers,
tokenization and embeddings, blocks, Q/K/V and attention, the MLP, output
probabilities, normalization/residuals/dropout, and all four pruning methods.
It paraphrases the Transformer Explainer material and explains exactly which
controls and measurements this local lab provides. Sampling concepts are
background reading, not additional prompt/temperature/top-k/top-p controls.

The model downloads on first use, so the first experiment can take longer.
Later runs reuse the local model cache. Read the execution safety section below
before consenting to run generated Python.

### Troubleshooting

- **Port 8000 is already in use:** choose another port:

  ```powershell
  & ..\.venv\Scripts\python.exe run_web.py --port 8001
  ```

  Then open **http://127.0.0.1:8001**.
- **Missing module / `ModuleNotFoundError`:** rerun the dependency installation
  command above, using the same virtual environment as the server.
- **Python executable not found:** complete the first-time setup and make sure
  PowerShell is open in the project folder.
- **Browser cannot connect:** check that the server is still running, inspect
  its terminal for errors, and use the same port shown in the startup output.
- **Invalid Python / `invalid syntax (<unknown>, line 1)` in older results:**
  check the selected model and the generated answer. GPT-2 can repeat the
  problem statement instead of producing Python; select **Qwen2.5-Coder · 1.5B
  Instruct** on the **Qwen experiments** page and start a new run. GPT-2 now has
  its own fixed sentence walkthrough and is no longer accepted by the coding
  experiment API. Existing saved result files are not rewritten.
- **Coding generation reached the context limit:** the answer is incomplete and will
  not be executed, even if part of it parses as Python. Use the coding model,
  shorten a custom problem, or reduce pruning strength. Failed generations
  still score 0; the app does not silently repair or replace model output.

## Page 1: fixed GPT-2 transformer walkthrough

Open the **GPT-2 walkthrough** menu tab. Its one fixed sentence is:

> the most pupulated country in the world is:

There is **no prompt editor**. The sentence, model, seed, and pruning settings
are owned by the server; modified request fields are rejected. The page opens
the prepared original/pruned comparisons automatically, without generation,
stop, history or export controls. Results are cached and reused across visits.
On a fresh installation, they are prepared once automatically; simultaneous
visits share the same preparation instead of starting duplicate GPU tasks.
The four predefined pruning examples use exactly that same sentence:

1. **Bypass the first block:** discard layer 1's contribution while preserving
   its input. This changes the actual model forward pass, not just its label.
2. **Mask one attention head per layer:** zero head 1's contribution in every
   layer before output projection, leaving the other eleven heads unmasked.
3. **Magnitude pruning:** zero 20% of the smallest-magnitude parameters in each
   targeted MLP tensor.
4. **Random pruning:** zero about 20% of MLP parameters using seed 42.

The Learn interface reuses the actual Svelte/D3 visualization components from
[Polo Club's Transformer Explainer](https://github.com/poloclub/transformer-explainer):
**embedding → Q/K/V → stacked self-attention → MLP → remaining blocks → softmax**.
This replaces the earlier hand-drawn approximation. It retains the upstream
stacked head cards, curved flow ribbons, residual operations, and expandable
embedding, attention and probability views, with pruning overlays added.

The upstream components receive this lab's local cached measurements, not a
second ONNX/browser model or the upstream cached examples. Original/pruned
measurements always come from the same Python model and fixed input. The page
does not contact the upstream website, CDNs, analytics, or external model APIs.
The MIT license and attribution are included in `frontend/LICENSE.upstream`
and `web/third-party/`; the pinned commit and changes are in `frontend/UPSTREAM.md`.

Select a compact pruning tab and toggle **Original / Pruned** to compare the
same layer and head in one diagram. Pruned heads have cut output routes,
bypassed blocks have a separate amber route, and the MLP weight sample marks
actual masked positions. Scroll up/down over **Layer** or **Head** to step through
them, or use their small arrow buttons. Previous/next-layer arrows above Q/K/V,
self-attention, and MLP keep the whole diagram synchronized.
Navigation stops at the first/last item and preserves horizontal panning.
Click embedding/probability headings or the attention matrix to expand the
upstream visualization in place. Concept/operation links open measured details
below it. Select an input token below the diagram to inspect its prediction.
Original/pruned text previews stay below the network; full continuations are
optional expandable details. Layer/head/token selection only changes inspection,
never the fixed input or applied presets. Small screens can pan the diagram
horizontally without making the page itself overflow.

The embeddings remain identical because these presets do not prune them.
Masked heads can retain the same attention matrix while their actual output
contribution is zero. Bypassed blocks still compute internally, but their
effective output equals their input. These distinctions are explicitly labeled.
The bypassed first layer is transparent with an orange dashed contour. When
inspecting a later layer, a small 3D layer-context card keeps layer 1 visible in
the background with the same orange dashed outline. Click that card to return
to layer 1. This layer-context card is separate from the attention-head stack.
The Original view never shows an applied bypass. The prepared-example version
has changed, so older last-layer or three-head snapshots are retained but not
reused as the first-layer/one-head lesson. The updated comparisons are prepared
once automatically.
The upstream decorative random vector textures have been replaced with measured
samples and shared original/pruned color scales. Unrecorded intermediate MLP
vectors are architectural placeholders, not invented activations. The expanded
probability view explicitly shows `log p`, not unavailable raw logits; displayed
probabilities retain their full-vocabulary values without top-k renormalization.

The upstream visualization is served in a local same-origin frame so its
styles do not affect the experiments page. Only that embedded document permits
the dynamic style attributes needed by Svelte/D3; the outer app retains its
strict content security policy. No inline or remote scripts are allowed.

### Rebuilding the visualization (frontend development only)

With Node.js 20.19+ or 22.12+:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

Commit both `frontend/` source changes and rebuilt `web/transformer/` assets.
The Python launcher remains unchanged. Restart the local Python server after
updating server code, then refresh the browser.

Measured comparisons appear after generation, not as a live network animation.
Snapshots describe the fixed input sentence, not every token of the growing
continuation. The last input token predicts the start of the continuation;
selecting an earlier input token inspects that earlier prefix.

GPT-2 uses raw text completion, greedy decoding, repetition penalty 1.1 and no
repeated 3-grams. Probability charts show the raw full-vocabulary distribution
before decoding controls; displayed top-eight probabilities are not renormalized.
GPT-2 is a small base model trained on older text, not a source of verified
current population rankings. No generated text
on this page is executed as Python or assigned a coding correctness score.

## Page 2: Qwen-only real-question experiments

Open **Qwen experiments** at `/experiments`. This is the existing coding lab,
now restricted to **Qwen2.5-Coder · 1.5B Instruct** in both the UI and API.

1. Select questions, review their tests, and choose pruning variants. You can
   edit problems or add custom math/coding exercises with a `solve(...)`
   contract and explicit expected test values.
2. Confirm restricted Python execution and click **Run comparison**.
3. Select a score cell or use **Tested question / Pruning variant**. The diagram,
   generated code, answer explorer, correctness and test outputs stay linked.

Both pages share a single model worker. Finish or stop an active task on one
page before starting the other. Each page shows only its own saved history;
old files are preserved, not silently relabeled as the fixed walkthrough.

### Qwen question-linked pipeline details

The pipeline includes:

- **Tokens and embeddings:** actual token IDs and decoded fragments, learned
  token embeddings and positional information. GPT-2 shows learned position
  vectors and their sums; Qwen correctly shows rotary positions on Q/K rather
  than inventing additive positional embeddings. Query selection is shared.
- **Self-attention:** choose any transformer layer and attention head; inspect
  colored token connections and QK-score, causal-mask, and softmax matrices.
  Select a matrix cell to inspect its query/key token pair.
- **Transformer computations:** measured Q/K/V samples, full-head scaled dot
  products, causal masking, attention-weighted head outputs, MLP outputs, and
  effective block input/output residual streams.
- **Next-token probabilities:** the top eight candidates and their actual
  full-vocabulary probabilities after the selected input token. The remaining
  probability mass is shown, not discarded or renormalized.
- **Question and tests:** the selected question's description, generated Python
  solutions, 0/1 outcome comparison, expected-test pass counts, and per-case
  inputs/outputs are shown with the diagram. Neural predictions are not test
  correctness scores.

These are bounded forward-pass measurements of the **exact prepared model
input used for the tested question**, including its generation prompt/chat
template. They are not a recording of every generated token or an animation
of the Python test runner. Longer inputs use **16 sampled prompt positions**
for display; the actual generation input is not shortened for visualization.
The interface shows original position indices, the full input token count,
and the exact model input. New oversized inputs are reported rather than
silently truncated; older saved truncation metadata is retained.

Attention is reconstructed from measured Q/K over **all causal keys**, then
sampled for display. Probability mass on unshown prompt tokens is reported,
not renormalized away. Vector heatmaps show the first eight dimensions; QK
scores use the full head dimension. Both models use shared color scales.
Masked heads still compute attention, but their measured output contributions
are zero. Bypassed blocks still compute internals, but their effective output
equals their input. These distinctions are labeled in the interface.

Coding generation and test scoring retain their original behavior. Probability
charts show raw model softmax values at temperature 1 for the next solution
token, **not** the probability that a Python test passes. A visualization
failure does not discard or alter solution/test results. Tensor samples are
loaded only for the selected question pair, rather than on every progress poll.

Interface concept inspired by Georgia Tech's
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/).
All displayed data comes from the local PyTorch model, not that website.
Existing history without question-linked forward-pass data cannot be
retroactively visualized; start a new question comparison to record it. Older
standalone text and GPT-2 coding result files remain on disk and readable via
the read-only results API, but do not appear as the new fixed walkthrough.

## Model and generation

Default: **`Qwen/Qwen2.5-Coder-1.5B-Instruct`**, coding-specialized, free
Apache 2.0 open weights. Approximately 3 GB FP16 weights fit an 8 GB laptop
GPU with additional room for generation. First use downloads the checkpoint;
later runs use the local cache. CUDA uses FP16; CPU uses FP32.

**There is no user-set generation token cap.** Generation ends at EOS,
cancellation, or the model's finite context limit (32,768 total tokens for
this Coder checkpoint; 1,024 for GPT-2). Context exhaustion is reported,
not silently treated as a complete answer. Severe pruning can cause long,
repetitive output. Use Stop if needed. Long generation consumes more time and KV-cache
memory. This cannot provide literally infinite generation.

GPT-2 is reserved for the fixed text walkthrough. Qwen is the only model for
new coding experiments. There is no artificial web generation token cap on
either page, but finite context windows cannot be removed by a UI switch.
Qwen has **32,768 total tokens** versus GPT-2's **1,024**. Oversized question
inputs are rejected with a clear message, not silently truncated. Long or
heavily pruned generations can still consume substantial time and memory.

## Coding problems

The default set contains ten original LeetCode-style exercises: Two Sum,
Valid Parentheses, Longest Common Prefix, Contains Duplicate, Maximum
Subarray, Binary Search, Merge Intervals, Move Zeroes, Climbing Stairs, and
Longest Substring Without Repeats. Each includes edge cases and expected
return values. Categories filter the local catalog of twenty exercises.

**New coding problem** selects an unused vetted problem from that catalog;
it does not ask an LLM to invent expected outputs. You can edit each problem
or add a custom problem as JSON. Always keep the description and tests in
agreement. Every solution must implement **`solve(*args)`**, returning its
answer rather than reading stdin or printing it. Tests are positional calls.

Example custom problem:

```json
{
  "id": "double",
  "title": "Double a number",
  "category": "arrays",
  "description": "Implement solve(n). Return twice n.",
  "tests": [
    {"args": [3], "expected": 6},
    {"args": [0], "expected": 0},
    {"args": [-4], "expected": -8}
  ],
  "normalize": "exact"
}
```

`normalize: "unordered"` sorts list/tuple return values before comparison;
this is used for Two Sum's index pair. Otherwise JSON return values are
compared exactly. Dictionary key order is ignored; list order and numeric
JSON types are significant. The current runner does not support linked-list
or tree object return values. Use JSON-compatible Python values instead.

## Binary comparison

For each problem, the unpruned and pruned programs run on **identical tests**:

- **1:** Both execute successfully and return equivalent outputs on every case.
- **0:** Any output differs, or either program is invalid, errors, or times out.

This is **output preservation, not correctness**. Two equally wrong programs
can score 1. The answer explorer separately shows how many expected tests
each program passes, the original/pruned code, errors, and the actual output
of every test. Summary percentages average only completed 0/1 comparisons.
Matching on these tests is not proof of equivalence for all possible inputs.
Runtime/performance differences are not part of the binary score.

## Restricted local execution: read before consenting

The UI requires explicit consent to execute generated Python. The server
never executes it in its own GPU/web process. Each solution gets a fresh
isolated child interpreter, empty temporary working directory, stripped
environment (no API credentials), a **5-second execution deadline**, and a
**256 MiB memory limit**. Windows uses Job Objects (including one-process and
kill-on-close limits); POSIX uses resource limits. Cancellation kills the
child. These execution budgets are independent of generation token length.

Defense-in-depth AST checks, restricted builtins and import allowlists block
normal file/network/subprocess/introspection access. Allowed algorithm
libraries include selected exports of `typing`, `collections`, `math`,
`heapq`, `bisect`, `itertools`, and `functools`. Some otherwise valid Python
features/imports are intentionally unavailable; such failures score 0 and
are explicitly reported. Debug prints are ignored; return values are scored.

**This is not a complete security sandbox.** Python language restrictions
are not a substitute for OS isolation. Use this local-only lab for trusted
experiments; do not expose it publicly or use it for hostile submissions.
For adversarial code use a hardened container/VM runner instead. Docker is
not installed on the current laptop and is not required by this local mode.

## Pruning and interface

- First-layer bypass in the fixed GPT-2 lesson; adjustable last-layer bypass in
  the Qwen experiments page. Per-layer head masking, magnitude MLP pruning,
  and random MLP pruning. The seed controls random masks; decoding is greedy.
- One cached model and one worker. Every variant is reverted in `finally`,
  including cancellation/failure. A model is discarded if restoration fails.
- Live progress/token counts, 0/1 problem-by-variant matrix, side-by-side code,
  per-case outputs/correctness, saved history, JSON export, and mobile layout.
- **GPT-2 walkthrough:** one locked sentence, four fixed pruning lessons,
  generated text, full-input attention matrices, and interactive final snapshots.
- **Qwen-Coder lab:** final question-linked pipeline snapshots with interactive
  question/variant/token/layer/head/stage selection, measured activations,
  attention reconstructed from measured Q/K, and next-token probabilities.
  Neither page uses invented measurements or live network animation.
- Masking does not remove dense parameters or guarantee speed/memory gains.
  Layer bypass still computes the block and then replaces its output.

Results save atomically to `results/web_runs/`. History lists
`python-outcomes-v1` coding runs (including question-linked pipelines) and
legacy `gpt2-pipeline-v1` text runs separately from old semantic-judge formats.
Legacy text runs have no Python correctness or binary equivalence score.
**Old result files are preserved** on disk.
Historical documentation/CLI examples are in `docs/LEGACY_README.md` and
`examples/`; they are not the coding app's execution path.

The server binds to 127.0.0.1 only, uses host validation and mutation tokens,
and runs as one worker (no reload/multiprocess). Model output renders as
escaped text under a restrictive CSP. Runs interrupted by server shutdown
are marked interrupted on restart, not automatically resumed.

## Tests

```powershell
& ..\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& ..\.venv\Scripts\python.exe -m pytest tests/ -q
```

Tests use tiny in-memory models and actual restricted Python child processes;
no model download, judge, or API charge is needed. They cover output matching,
wrong-but-equal results, errors, forbidden imports, timeout/memory/cancellation,
web validation/consent, persistence and actual pruning restoration.

Optional browser regression checks cover scrolling through layers/heads,
diagram arrows, keyboard focus, boundary limits, and mobile panning:

```powershell
& ..\.venv\Scripts\python.exe -m pip install playwright
# Skip the next line if Microsoft Edge is installed.
& ..\.venv\Scripts\python.exe -m playwright install chromium
& ..\.venv\Scripts\python.exe -m pytest tests/test_learn_navigation.py -q
```

These checks use synthetic test fixtures, not generated model measurements.

See `CODE_REVIEW.md` for the earlier review and remaining legacy structural
pruning limitations. An API credential removed in that review should still
be revoked/rotated even though this coding app no longer uses credentials.
