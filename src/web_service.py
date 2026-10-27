"""Single-worker experiment service. Only this worker touches the cached model."""

import copy
import gc
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer

from .judge_eval import get_model_answer
from .coding_problems import Problem
from .python_runner import run_solution, compare_outcomes
from .model_loader import DEFAULT_MODEL_NAME, load_model_and_tokenizer, get_model_metadata
from .pruners.head_pruner import HeadPruner
from .pruners.module_pruner import ModulePruner
from .pruners.weight_pruner import WeightPruner
from .visualization import gpt2_snapshot
from .text_pipeline import capture_pipeline, prompt_inputs
from .question_pipeline import capture_question_pipeline
from .walkthrough import WALKTHROUGH_VERSION, WALKTHROUGH_PRESETS
from .pruning_targets import bypassed_layer_indices


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technique: Literal["layers", "heads", "magnitude", "random"]
    strength: int = Field(ge=1, le=80)
    layer_position: Literal["first", "last"] = "last"

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        # Preserve existing last-layer experiment configs and saved records.
        if self.layer_position == "last":
            data.pop("layer_position", None)
        return data

    @model_validator(mode="after")
    def valid_layer_position(self):
        if self.layer_position == "first" and self.technique != "layers":
            raise ValueError("Layer position applies only to layer bypass.")
        return self

    @field_validator("strength")
    @classmethod
    def valid_strength(cls, value, info):
        limit = {"layers": 27, "heads": 11}.get(info.data.get("technique"), 80)
        if value > limit:
            raise ValueError(f"Strength exceeds {limit} for this technique.")
        return value

    @property
    def label(self):
        if self.technique == "layers":
            return f"Bypass {self.layer_position} {self.strength} layer(s)"
        if self.technique == "heads":
            return f"Mask {self.strength}/12 heads per layer"
        return f"MLP {self.technique} · {self.strength}%"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(default="Python coding", min_length=1, max_length=160)
    model_name: Literal["Qwen/Qwen2.5-Coder-1.5B-Instruct", "gpt2"] = DEFAULT_MODEL_NAME
    problems: list[Problem] = Field(min_length=1, max_length=20)
    variants: list[Variant] = Field(min_length=1, max_length=8)
    execution_consent: bool = False
    seed: int = Field(default=42, ge=0, le=2147483647)

    @model_validator(mode="after")
    def model_bounds(self):
        if self.model_name == "gpt2" and any(v.technique == "layers" and v.strength > 11 for v in self.variants):
            raise ValueError("GPT-2 has 12 layers; bypass at most 11.")
        if not self.execution_consent:
            raise ValueError("Confirm local restricted Python execution before starting.")
        if len({p.id for p in self.problems}) != len(self.problems):
            raise ValueError("Remove duplicate problems.")
        return self


class Cancelled(Exception):
    pass


class TextRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: Literal["gpt2"] = "gpt2"
    prompt: str = Field(min_length=1, max_length=1500)
    variants: list[Variant] = Field(min_length=1, max_length=8)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @model_validator(mode="after")
    def valid_prompt_and_variants(self):
        if not self.prompt.strip():
            raise ValueError("Enter some text for GPT-2 to continue.")
        if any(v.technique == "layers" and v.strength > 11 for v in self.variants):
            raise ValueError("GPT-2 has 12 layers; bypass at most 11.")
        return self

    @property
    def topic(self):
        return "GPT-2 · " + self.prompt[:80]


class Runtime:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.on_token = None
        self._active_pruner = None

    def load(self, model_name=DEFAULT_MODEL_NAME):
        if self.model is None or self.model_name != model_name:
            # Only one resident model; model selection/judging never stacks GPU copies.
            self.model = self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.set_num_threads(4)
            self.model, self.tokenizer = load_model_and_tokenizer(model_name)
            self.model_name = model_name

    def answer(self, question, tokens, cancel):
        if cancel.is_set():
            raise Cancelled()
        started = time.perf_counter()
        answer = get_model_answer(self.model, self.tokenizer, question,
                                  max_new_tokens=tokens, cancel_event=cancel, return_details=True,
                                  on_token=self.on_token)
        if cancel.is_set():
            raise Cancelled()
        return {**answer, "seconds": round(time.perf_counter() - started, 3)}

    def baseline_visualization(self):
        return gpt2_snapshot(self.model, None, None)

    def question_pipeline(self, problem, cancel, variant=None):
        if cancel.is_set():
            raise Cancelled()
        pipeline = capture_question_pipeline(self.model, self.tokenizer, problem, variant, self._active_pruner)
        if cancel.is_set():
            raise Cancelled()
        return pipeline

    def text_answer(self, prompt, cancel, topology):
        if cancel.is_set():
            raise Cancelled()
        # Validate before generation rather than silently truncating the prompt.
        prompt_inputs(self.model, self.tokenizer, prompt)
        pipeline = capture_pipeline(self.model, self.tokenizer, prompt, topology)
        if cancel.is_set():
            raise Cancelled()
        started = time.perf_counter()
        answer = get_model_answer(
            self.model, self.tokenizer, prompt, max_new_tokens=None,
            cancel_event=cancel, return_details=True, on_token=self.on_token,
            raw_prompt=True, repetition_penalty=1.1, no_repeat_ngram_size=3,
        )
        if cancel.is_set():
            raise Cancelled()
        return {**answer, "seconds": round(time.perf_counter() - started, 3), "pipeline": pipeline}

    @contextmanager
    def pruned(self, variant, seed):
        metadata = get_model_metadata(self.model)
        prefix = "transformer.h" if metadata["model_type"] == "gpt2" else "model.layers"
        if variant.technique == "layers":
            pruner = ModulePruner(self.model, {
                "target_modules": [f"{prefix}.{i}" for i in bypassed_layer_indices(variant, metadata["num_layers"])],
                "masking_replacement": "identity",
            })
        elif variant.technique == "heads":
            count = min(variant.strength, metadata["num_heads"] - 1)
            pruner = HeadPruner(self.model, {"head_indices": list(range(count))})
        else:
            pruner = WeightPruner(self.model, {
                "strategy": variant.technique, "sparsity": variant.strength / 100,
                "target_modules": [f"{prefix}.*.mlp.*"],
            })
        # Isolate random masking from unrelated RNG use. All requests are serialized.
        devices = [self.model.device.index or 0] if self.model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            try:
                pruner.apply_masking()
                self._active_pruner = pruner
                yield gpt2_snapshot(self.model, variant, pruner)
            finally:
                self._active_pruner = None
                try:
                    pruner.revert()
                except Exception:
                    # Never reuse a possibly contaminated model after failed restoration.
                    self.model = self.tokenizer = None
                    raise


class ExperimentService:
    def __init__(self, output_dir=None, runtime=None):
        self.output_dir = Path(output_dir or Path(__file__).resolve().parents[1] / "results" / "web_runs")
        self.runtime = runtime or Runtime()
        self.lock = threading.RLock()
        self.active = None
        self.cancel_event = threading.Event()
        self.runs = {}
        if self.output_dir.exists():
            for file in sorted(self.output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)[-50:]:
                try:
                    data = json.loads(file.read_text(encoding="utf-8"))
                    uuid.UUID(data["id"])
                    if data["status"] in ("running", "queued", "cancelling"):
                        data["status"] = "interrupted"
                        data["stage"] = "Server restarted. Partial results retained."
                    if data.get("evaluation") in ("python-outcomes-v1", "gpt2-pipeline-v1"):
                        self.runs[data["id"]] = data
                except (OSError, ValueError, KeyError, TypeError):
                    continue

    def snapshot(self, run_id, include_pipelines=True):
        with self.lock:
            record = self.runs[run_id]
            if include_pipelines:
                return copy.deepcopy(record)
            # Progress polling must not repeatedly copy/send every large tensor
            # sample for all questions. Selected pairs are requested separately.
            def compact(answer):
                result = {key: value for key, value in answer.items() if key != "pipeline"}
                result["pipeline_available"] = bool(answer.get("pipeline"))
                return result
            result = {**record, "baselines": {key: compact(answer) for key, answer in record["baselines"].items()},
                      "comparisons": {key: {**cell, "answer": compact(cell["answer"])} for key, cell in record["comparisons"].items()}}
            return copy.deepcopy(result)

    def pipeline_pair(self, run_id, question, variant):
        with self.lock:
            record = self.runs[run_id]
            if record["status"] in ("running", "queued", "cancelling"):
                raise RuntimeError("Pipeline inspection is available after the run finishes.")
            count = len(record["config"].get("problems", [None]))
            if not 0 <= question < count or not 0 <= variant < len(record["config"]["variants"]):
                raise IndexError("Question or variant is outside this run.")
            baseline = record["baselines"].get(str(question), {})
            pruned = record["comparisons"].get(f"{variant}:{question}", {}).get("answer", {})
            return copy.deepcopy({"question": question, "variant": variant,
                                  "baseline": baseline.get("pipeline"), "pruned": pruned.get("pipeline")})

    def history(self):
        with self.lock:
            return [{**{k: r[k] for k in ("id", "topic", "status", "created_at", "kind")},
                     "model": r.get("model"), "walkthrough_version": r.get("walkthrough_version")}
                    for r in sorted(self.runs.values(), key=lambda r: r["created_at"], reverse=True)
                    if r["kind"] in ("experiment", "text-generation", "walkthrough")]

    def update(self, run_id, **values):
        with self.lock:
            self.runs[run_id].update(values)
            self.runs[run_id]["revision"] += 1

    def save(self, run_id):
        data = self.snapshot(run_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{run_id}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def ensure_walkthrough(self, request):
        """Reuse a ready example or an in-flight preparation, never duplicate it.

        Cached results can be read even while the shared worker is busy with
        Qwen. All clients share the same prepared walkthrough for this version.
        """
        config = request.model_dump()
        with self.lock:
            def matches(record):
                return (record.get("kind") == "walkthrough"
                        and record.get("walkthrough_version") == WALKTHROUGH_VERSION
                        and record.get("config") == config)

            for record in sorted(self.runs.values(), key=lambda run: run["created_at"], reverse=True):
                if (matches(record) and record["status"] == "completed"
                        and record.get("baselines", {}).get("0", {}).get("pipeline")
                        and all(record.get("comparisons", {}).get(f"{index}:0", {}).get("answer", {}).get("pipeline")
                                for index in range(len(request.variants)))):
                    return record["id"]
            active = self.runs.get(self.active, {})
            if matches(active) and active.get("status") in ("queued", "running", "cancelling"):
                return self.active
            return self.start(request, "walkthrough")

    def start(self, request, kind="experiment"):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("The model is busy. Stop or wait for the current task.")
            run_id = str(uuid.uuid4())
            record = {
                "id": run_id, "kind": kind, "topic": request.topic, "created_at": time.time(),
                "status": "queued", "stage": "Waiting for model", "revision": 0,
                "done": 0, "total": 1, "error": None, "model": getattr(request, "model_name", DEFAULT_MODEL_NAME),
                "config": request.model_dump(), "baselines": {}, "comparisons": {},
                "evaluation": "gpt2-pipeline-v1" if isinstance(request, TextRunRequest) else "python-outcomes-v1",
                "phase": "loading", "current_variant": None, "current_question": None,
                "generated_tokens": 0, "mask_active": False, "pruning_views": {},
            }
            if kind == "walkthrough":
                record["walkthrough_version"] = WALKTHROUGH_VERSION
                record["walkthrough_presets"] = copy.deepcopy(list(WALKTHROUGH_PRESETS))
                record["topic"] = "GPT-2 · fixed transformer walkthrough"
            self.runs[run_id] = record
            self.active = run_id
            self.cancel_event = threading.Event()
            self.worker = threading.Thread(target=self._worker, args=(run_id, request, kind), daemon=True)
            self.worker.start()
            return run_id

    def cancel(self, run_id):
        with self.lock:
            if self.active == run_id:
                self.cancel_event.set()
                self.update(run_id, status="cancelling", stage="Stopping safely and restoring weights…")

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled()

    def _execute_answer(self, answer, problem, model_name):
        execution = run_solution(answer["text"], problem, self.cancel_event,
                                 truncated=answer.get("truncated", False))
        if execution["status"] == "invalid" and model_name == "gpt2":
            execution["error"] += (
                " GPT-2 is not coding/instruction tuned and may repeat the prompt instead of writing Python. "
                "Select Qwen2.5-Coder for useful coding solutions. This experimental GPT-2 run still uses the same questions and tests."
            )
        return execution

    def _record_question_pipeline(self, answer, problem, variant=None):
        capture = getattr(self.runtime, "question_pipeline", None)
        if capture is None:
            return
        self.check_cancel()
        try:
            answer["pipeline"] = capture(problem, self.cancel_event, variant)
        except Cancelled:
            raise
        except Exception as exc:
            # Visualization failures must not replace or invalidate test outcomes.
            answer["pipeline_error"] = f"Pipeline unavailable ({type(exc).__name__}); the generated solution and test results are retained."
            import traceback
            traceback.print_exc()
        self.check_cancel()

    def _worker(self, run_id, request, kind):
        try:
            self.update(run_id, status="running", stage="Loading local model…")
            self.runtime.on_token = lambda tokens: self.update(run_id, generated_tokens=tokens)
            self.runtime.load(request.model_name)
            self.check_cancel()
            self.update(run_id, baseline_view=getattr(self.runtime, "baseline_visualization", lambda: None)())
            if isinstance(request, TextRunRequest):
                self._text_experiment(run_id, request)
            else:
                self._experiment(run_id, request)
            self.check_cancel()
            self.update(run_id, status="completed", phase="complete", mask_active=False, stage="Complete")
        except Cancelled:
            self.update(run_id, status="cancelled", phase="cancelled", mask_active=False, stage="Stopped. Original weights restored.")
        except Exception as exc:
            # Raw provider response bodies, headers and credentials are never stored.
            message = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: task failed; check the server console."
            self.update(run_id, status="failed", phase="failed", mask_active=False, stage="Task failed", error=message)
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.save(run_id)
            except OSError:
                self.update(run_id, error="Could not save results to disk. Export JSON from this page.")
            with self.lock:
                self.active = None

    def _text_experiment(self, run_id, request):
        self.update(run_id, total=1 + len(request.variants), phase="baseline",
                    stage="Generating original text and recording its pipeline…")
        baseline_view = self.snapshot(run_id).get("baseline_view")
        baseline = self.runtime.text_answer(request.prompt, self.cancel_event, baseline_view)
        self.check_cancel()
        self.update(run_id, baselines={"0": baseline}, done=1)
        self.save(run_id)
        comparisons, views = {}, {}
        for index, variant in enumerate(request.variants):
            self.check_cancel()
            self.update(run_id, phase="pruned", current_variant=index, generated_tokens=0,
                        stage=f"Generating text · {variant.label}")
            with self.runtime.pruned(variant, request.seed) as topology:
                self.update(run_id, mask_active=True)
                answer = self.runtime.text_answer(request.prompt, self.cancel_event, topology)
                self.check_cancel()
                comparisons[f"{index}:0"] = {"answer": answer, "status": "completed"}
                views[str(index)] = topology
            # Publish only restored, finished states, not a live activation stream.
            self.update(run_id, comparisons=copy.deepcopy(comparisons), pruning_views=copy.deepcopy(views),
                        done=index + 2, mask_active=False)
            self.save(run_id)

    def _experiment(self, run_id, request):
        count = len(request.problems)
        self.update(run_id, total=count * (1 + len(request.variants)))
        baselines, comparisons = {}, {}
        done = 0
        for qi, problem in enumerate(request.problems):
            self.check_cancel()
            self.update(run_id, stage=f"Baseline answer {qi + 1}/{count}", phase="baseline",
                        current_question=qi, generated_tokens=0)
            answer = self.runtime.answer(problem.prompt(), None, self.cancel_event)
            self.update(run_id, phase="testing", stage=f"Testing original solution {qi + 1}/{count}")
            answer["execution"] = self._execute_answer(answer, problem, request.model_name)
            self.update(run_id, stage=f"Recording pipeline for original question {qi + 1}/{count}")
            self._record_question_pipeline(answer, problem)
            self.check_cancel()
            baselines[str(qi)] = answer
            done += 1
            self.update(run_id, baselines=copy.deepcopy(baselines), done=done)
            self.save(run_id)

        for vi, variant in enumerate(request.variants):
            self.check_cancel()
            self.update(run_id, phase="applying", current_variant=vi, generated_tokens=0,
                        stage=f"Applying {variant.label}…")
            self.runtime.load(request.model_name)
            with self.runtime.pruned(variant, request.seed) as view:
                if view is not None:
                    views = self.snapshot(run_id)["pruning_views"]
                    views[str(vi)] = view
                    self.update(run_id, pruning_views=views)
                self.update(run_id, mask_active=True)
                for qi, problem in enumerate(request.problems):
                    self.check_cancel()
                    self.update(run_id, stage=f"{variant.label} · answer {qi + 1}/{count}",
                                phase="pruned", current_question=qi, generated_tokens=0)
                    answer = self.runtime.answer(problem.prompt(), None, self.cancel_event)
                    self.update(run_id, phase="testing", stage=f"Executing pruned solution {qi + 1}/{count}")
                    execution = self._execute_answer(answer, problem, request.model_name)
                    self.check_cancel()
                    answer["execution"] = execution
                    self.update(run_id, stage=f"Recording pipeline for pruned question {qi + 1}/{count}")
                    self._record_question_pipeline(answer, problem, variant)
                    verdict = compare_outcomes(baselines[str(qi)]["execution"], execution)
                    comparisons[f"{vi}:{qi}"] = {
                        "answer": answer, "status": "completed", **verdict,
                    }
                    done += 1
                    self.update(run_id, comparisons=copy.deepcopy(comparisons), done=done)
                    self.save(run_id)

                self.update(run_id, phase="restoring", stage="Restoring original weights…")

            self.update(run_id, mask_active=False)
