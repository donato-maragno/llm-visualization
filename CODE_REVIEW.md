# Code review and interface implementation

## Upstream visualization integration

The Learn page now embeds a local build of the actual Transformer Explainer
Svelte/D3 components, with measured pruning overlays and no application logo.
The Python pruning backend and separate Qwen page are unchanged. Source,
pinned upstream commit, MIT notice and rebuild instructions are in `frontend/`.
The compiled assets are included so Python users do not need Node/npm.

The upstream ONNX runtime, cached examples and analytics are not included.
Decorative random vector textures are replaced by actual measured samples.
The outer page retains its strict CSP; only the same-origin visualization
document allows dynamic style attributes and local/data fonts needed by the
upstream libraries. Script execution remains restricted to local assets.
Browser regression tests cover real upstream components, original/pruned
data, all four pruning presets, navigation, keyboard interaction, expansion
overflow, frame-origin validation, mobile panning, and absence of script/CSP errors.

The fixed GPT-2 bypass lesson now targets the first transformer layer in the
actual forward pass, with a transparent orange dashed contour. On later layers,
a separate 3D layer-context card retains the outlined first layer behind the
selected layer and links back to it. Qwen's default last-N-layer experiments
remain unchanged. Shared layer selection keeps runtime targets and telemetry
consistent; the lesson version invalidates old last-layer cached comparisons
without modifying those saved results. Validation: 147 tests passed, plus a
live GPT-2/browser check of first-layer identity bypass and later-layer context.

The Learn head-masking preset now zeros just head 1 in each layer; the other
eleven are unmasked. A versioned prepared example prevents older three-head
comparisons from being relabeled. A static nine-chapter guide follows the
visualization, with paraphrased architecture explanations, all four pruning
methods, inference/training distinctions, and accurate descriptions of this
lab's controls. General sampling concepts are not presented as enabled controls.
Validation: 151 tests passed, including guide anchors/mobile layout and an
actual GPT-2 check of zeroed head-1 outputs and nonzero head-2 outputs.

The older review notes and validation counts below describe previous iterations.

## Current coding-only workflow

The app and `run_all.py` now use Python LeetCode-style problems and
`Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache 2.0), not semantic judging.
No judge/API settings are accepted by the new run schema. The ten default
problems include explicit tests; twenty vetted exercises are available.
Custom problems must supply tests and a solve(...) contract. The new
`python-outcomes-v1` history is separated from legacy semantic-score files,
which are preserved rather than reinterpreted as binary outcomes.

Generation has no application-selected token cap: it stops at EOS, cancel,
or the actual model context limit. Execution budgets are separate. The
restricted local runner uses a stripped child environment, AST/import/
builtin restrictions, a five-second timeout and 256 MiB memory cap. Windows
Job Objects enforce process/memory limits; POSIX uses resource limits.
Execution requires explicit consent. **This is defense in depth, not a
complete sandbox for hostile Python.** Use OS/container isolation for that.

Binary preservation is 1 only when both programs execute and return the
same normalized JSON outcomes on every provided case; otherwise it is 0.
Two wrong-but-equal programs can score 1, so expected-test correctness is
displayed separately. Identical errors are never treated as matches.
The catalog is deliberately limited to JSON arguments/returns, not arbitrary
linked-list/tree objects or user-selected entry points.

The GPT-2 visualization now uses connected SVG topology, residual routes,
explicit cut marks and amber layer-bypass paths rather than numbered head
boxes. Teal/violet signed colors use actual weight samples, not random
decoration. Numerical details are optional; the view is explicitly not an
activation trace. Unit tests compare telemetry against actual parameters.

Current automated validation: **58 passing tests**, including real child
process execution, correct/wrong-but-equal outcomes, error handling, blocked
imports, timeouts/memory limits, cancellation, consent and API validation,
pruning restoration and visual telemetry. Real browser testing also covers
the Coder model's ten-problem 0/1 workflow and the redesigned live GPT-2
pathways/weight-color view. Historical review findings below remain useful
for legacy files; standalone judge examples are not used by the app.

---

Reviewed the Python source, pruning classes, configuration loader/YAML,
example entry points, tests, requirements and documentation in this folder.
Existing checkpoint binaries and cached/generated artifacts were not
deserialized or replaced as part of the review.

## Findings addressed

| Priority | Finding | Resolution |
|---|---|---|
| Critical | An OpenRouter credential was embedded in `src/judge_eval.py`. | Removed. Read credentials from the server environment only. **Rotate the exposed key.** |
| High | API failures and malformed judge replies silently became similarity `0`. | Strict JSON/score validation; errors propagate in the CLI and are explicitly shown/excluded in the UI. Explanations are retained by the detailed judge API. |
| High | Sampled answers could differ even with no pruning. | Greedy decoding for comparisons; explicit seed for random weight masks. |
| High | GPU/model mutation could overlap in an interactive server. | One cached model, one background worker, busy-request rejection, restoration in `finally`, and discard-on-restore-failure. No multi-worker/reload server. |
| High | Entire-layer zeroing destroys the residual stream, unlike removing a layer. | UI uses explicit residual-preserving identity bypass, not the older raw-zero mode. |
| Medium | `config/config_loader.py` imported nonexistent `config.pruners`. | Factory imports now resolve to `src.pruners`. |
| Medium | Legacy distribution metrics used FP16 probability ratios that underflowed, included the wrong prediction positions, and double-counted overlap. | FP32 log-softmax/log-add-exp, correct target positions, and one contribution per next-token target. Tests cover extreme logits and overlapping windows. |
| Medium | Perplexity weighted input counts rather than new predicted targets. | Mask overlap labels and weight losses by actual new target counts. |
| Medium | Token-fidelity output labeled logits as probabilities. | Use softmax probabilities at the selected token indices. |
| Medium | GPU latency timings lacked explicit synchronization and assumed every generation reached the token cap. | Synchronize timed CUDA work; compute throughput from actual generated lengths. |
| Medium | Generic tuple hooks attempted `zeros_like(None)` and could modify cache metadata. | Zero/scale only the hidden-state item; preserve auxiliary outputs. |
| Medium | The standalone head-mask helper still masked features after output projection. | Share the pre-projection head mask used by `HeadPruner`; clone inputs before masking. |
| Medium | Local judge prompt contained an illustrative numeric score, risking anchoring. | Removed the example number. UI flags uniform scores and labels the local self-judge as uncalibrated. |
| Low | Answer truncation was not visible. | Preserve token count/cap metadata and show “token cap reached” alongside answer timing. |

## Remaining limitations / deliberately excluded from the web UI

- **Legacy structural pruning is not deployment-ready.** In particular,
  decoder-layer replacements in `ModulePruner`/`structural.py` assume older
  tuple return signatures, while recent Qwen2 blocks return tensors. The
  standalone structural helpers also lack consistent Qwen layer discovery.
  Do not use them for cached Qwen generation without architecture-specific
  tests and cache handling. The UI never calls these paths.
- `HeadPruner.apply_structural` is weight zeroing, not actual dimension
  removal. Fused GPT-2 QKV slicing and grouped-query shared KV behavior need
  additional architecture-specific corrections/tests before using that path.
  The UI exposes pre-projection **masking** only; it does not zero shared KV
  tensors or claim structural speedups.
- Old CLI scripts often load two copies of the model, lack `try/finally`
  around their whole experiment, or use raw output zeroing. Prefer the UI or
  local benchmark for reversible single-model experiments. These scripts
  were not comprehensively rewritten, preserving their existing behavior.
- There is no training/recovery phase, importance-ranked head selection,
  representative benchmark corpus, calibrated judge, or statistical
  uncertainty estimate. A small-model self-judge is useful for inspection,
  **not a scientific quality measurement**. Meaning preservation does not
  imply correctness. Review the two answers and the explanation yourself.
- Zeroed dense weights and layer bypass hooks do not shrink checkpoints or
  guarantee speed/memory savings. FP16 magnitude ties make sparsity slightly
  lower than its requested threshold. Percentages in the UI apply to MLPs,
  not the whole model; N heads refers to each layer, not the model globally.
- The server is local and single-user. State is process-local, while result
  snapshots are persisted atomically. It loads the most recent 50 saved
  records at startup. No job resume after process termination; partial work
  is marked interrupted. Do not run multiple worker processes or expose it
  publicly without authentication, stronger resource limits and an external
  job queue. External judge calls can take up to their 120-second timeout
  before cancellation completes.

## Validation

- 53 automated tests passing: existing pruning smoke tests plus API input
  bounds, mutation-token protection, persistence/restart behavior, strict
  judge parsing, failure/cancellation, one-worker enforcement, real tiny-Qwen
  mask restoration, metrics and configuration-factory regressions.
- Microsoft Edge/Playwright real-browser test: ten questions, layer bypass
  and MLP magnitude variants, baseline/pruned generation and local judging;
  overview-to-answer navigation, question generation, custom question entry
  and 390px mobile layout without horizontal document overflow.
- Actual Qwen2.5-1.5B FP16 inference verified on the RTX 4060 Laptop GPU.
  OpenRouter was **not called** during verification; its HTTP contract is
  tested with mock responses, not live paid calls.
- Final ten-question browser run completed all 20 local judgments with
  valid JSON scores and explanations. Scores ranged from 0.0 to 0.95 after
  removing the anchoring example. The smoke test used a 64-token cap and
  therefore truncated answers; it verifies operation, not answer quality.
- GPT-2-only visualization tested with real GPT-2 inference in Microsoft
  Edge: live layer-bypass, head-output and MLP-mask states, original-state
  restoration, recorded-mask inspection, and hiding the visual panel for
  Qwen. Tiny GPT-2 tests compare telemetry to actual zeroed positions and
  verify exact restoration for all four techniques. Model-specific input
  bounds and use of unpruned Qwen (not GPT-2) for local judging are tested.

The Python web server needs no Node build step, CDN dependencies, remote fonts, or
hard-coded keys. Results render as escaped/plain text (not model-provided
HTML), and a restrictive content security policy is applied.
