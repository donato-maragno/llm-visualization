"""Measured text pipelines: tiny local models only, no downloads or Python execution."""

import json
import threading
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

from src.judge_eval import get_model_answer
from src.text_pipeline import capture_pipeline, prompt_inputs
from src.web_service import ExperimentService, Runtime, TextRunRequest, Variant
from web_app import create_app


class TinyTokenizer:
    chat_template = None
    eos_token_id = 31

    def __init__(self):
        self.prompts = []

    def __call__(self, text, **kwargs):
        assert kwargs.get("truncation") is False
        self.prompts.append(text)
        ids = torch.tensor([[1 + ord(c) % 29 for c in text]])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, **kwargs):
        return "".join(f"t{int(i)}" for i in ids)


@pytest.fixture
def runtime():
    torch.manual_seed(7)
    runtime = Runtime()
    runtime.model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=64, n_embd=32, n_head=4, n_layer=2,
        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, eos_token_id=31, bos_token_id=31,
    )).eval()
    runtime.tokenizer = TinyTokenizer()
    return runtime


def test_forward_snapshot_is_measured_and_json_safe(runtime):
    model, tokenizer = runtime.model, runtime.tokenizer
    implementation = model.config._attn_implementation
    view = capture_pipeline(model, tokenizer, "Data")
    json.dumps(view, allow_nan=False)
    assert model.config._attn_implementation == implementation
    assert view["schema"] == "gpt2-forward-v1"
    assert len(view["tokens"]) == len(view["predictions"]) == 4
    inputs = prompt_inputs(model, tokenizer, "Data")
    with torch.inference_mode():
        expected = model(**inputs).logits[0].float().softmax(-1)
        token = model.transformer.wte(inputs.input_ids[0])[:, :8]
        position = model.transformer.wpe(torch.arange(4))[:, :8]
    assert torch.allclose(torch.tensor(view["embeddings"]["combined"]), token + position, atol=1e-6)
    for i, prediction in enumerate(view["predictions"]):
        for candidate in prediction["candidates"]:
            assert candidate["probability"] == pytest.approx(expected[i, candidate["id"]].item(), abs=1e-6)
        assert sum(c["probability"] for c in prediction["candidates"]) + prediction["other_probability"] == pytest.approx(1)
    for layer in view["layers"]:
        attention = torch.tensor(layer["attention"])
        scores = torch.tensor(layer["scores"])
        scores.masked_fill_(torch.ones(4, 4, dtype=torch.bool).triu(1), float("-inf"))
        assert torch.allclose(scores.softmax(-1), attention, atol=2e-6)
        assert torch.allclose(attention.sum(-1), torch.ones(4, 4), atol=2e-6)
        assert torch.count_nonzero(attention.triu(1)) == 0
        assert not model.transformer.h[layer["index"]]._forward_hooks
        assert not model.transformer.h[layer["index"]].attn.c_attn._forward_hooks


@pytest.mark.parametrize("technique,strength", [("layers",1),("heads",1),("magnitude",35),("random",50)])
def test_pipeline_shows_actual_mask_effects_and_restores(runtime, technique, strength):
    original = capture_pipeline(runtime.model, runtime.tokenizer, "Data")
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    with runtime.pruned(Variant(technique=technique, strength=strength), 42) as topology:
        pruned = capture_pipeline(runtime.model, runtime.tokenizer, "Data", topology)
        assert pruned["tokens"] == original["tokens"]
        if technique == "heads":
            for layer in pruned["layers"]:
                assert layer["masked_heads"] == [0]
                assert not torch.count_nonzero(torch.tensor(layer["head_output"][0]))
                assert torch.count_nonzero(torch.tensor(layer["attention"][0]))
        elif technique == "layers":
            assert pruned["layers"][-1]["bypassed"]
            assert pruned["layers"][-1]["input"] == pruned["layers"][-1]["output"]
        else:
            assert all(layer["masked_weights"] > 0 for layer in pruned["layers"])
        assert pruned["predictions"] != original["predictions"]
    for name, value in runtime.model.state_dict().items():
        assert torch.equal(value, before[name])
    restored = capture_pipeline(runtime.model, runtime.tokenizer, "Data")
    assert restored["predictions"] == original["predictions"]


def test_capture_failure_removes_only_its_own_hooks(runtime, monkeypatch):
    implementation = runtime.model.config._attn_implementation
    with runtime.pruned(Variant(technique="heads", strength=1), 42):
        projection = runtime.model.transformer.h[0].attn.c_proj
        previous = dict(projection._forward_pre_hooks)
        def fail(*args, **kwargs):
            raise RuntimeError("Forward failed")
        monkeypatch.setattr(runtime.model, "forward", fail)
        with pytest.raises(RuntimeError, match="Forward failed"):
            capture_pipeline(runtime.model, runtime.tokenizer, "Data")
        assert dict(projection._forward_pre_hooks) == previous
        assert not runtime.model.transformer.h[0].attn.c_attn._forward_hooks
        assert runtime.model.config._attn_implementation == implementation


def test_long_prompt_is_rejected_without_truncation(runtime):
    with pytest.raises(ValueError, match="No input was truncated"):
        capture_pipeline(runtime.model, runtime.tokenizer, "x" * 33)


def test_gpt2_raw_completion_preserves_prompt_and_leading_space():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=64)
            self.generation_config = SimpleNamespace(eos_token_id=31)
        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 60
            assert kwargs["repetition_penalty"] == 1.1
            assert kwargs["no_repeat_ngram_size"] == 3
            return torch.cat((kwargs["input_ids"], torch.tensor([[5, 31]])), dim=-1)
    tokenizer = TinyTokenizer()
    tokenizer.decode = lambda *a, **kw: " continue"
    answer = get_model_answer(Model(), tokenizer, "Data", raw_prompt=True,
                              max_new_tokens=None, return_details=True,
                              repetition_penalty=1.1, no_repeat_ngram_size=3)
    assert tokenizer.prompts == ["Data"]
    assert answer["text"] == " continue"
    assert answer["stop_reason"] == "eos" and not answer["truncated"]


class FakeTextRuntime:
    def __init__(self, fail=False):
        self.pruning = False
        self.restored = False
        self.fail = fail
        self.on_token = None
    def load(self, model_name):
        assert model_name == "gpt2"
    def text_answer(self, prompt, cancel, topology):
        assert not cancel.is_set()
        if self.fail and self.pruning:
            raise ValueError("Text generation failed")
        return {"text": " build better tools.", "tokens": 5, "seconds": .1,
                "truncated": False, "stop_reason": "eos", "pipeline": {"prompt": prompt}}
    def pruned(self, variant, seed):
        from contextlib import contextmanager
        @contextmanager
        def context():
            self.pruning = True
            try:
                yield {"technique": variant.technique}
            finally:
                self.pruning = False
                self.restored = True
        return context()


def text_request(**overrides):
    return TextRunRequest(prompt="The sum of two numbers is",
                          variants=[Variant(technique="heads", strength=1)], **overrides)


def finish(service, run_id):
    service.worker.join(timeout=10)
    record = service.snapshot(run_id)
    assert record["status"] not in ("queued", "running", "cancelling")
    return record


def test_text_lifecycle_has_no_python_execution_and_persists(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("GPT-2 text generation must never invoke the Python runner")
    monkeypatch.setattr("src.web_service.run_solution", forbidden)
    runtime = FakeTextRuntime()
    service = ExperimentService(tmp_path, runtime)
    record = finish(service, service.start(text_request(), "text-generation"))
    assert record["status"] == "completed"
    assert record["evaluation"] == "gpt2-pipeline-v1"
    assert record["done"] == record["total"] == 2
    assert runtime.restored and not runtime.pruning
    assert "execution" not in record["baselines"]["0"]
    assert "score" not in record["comparisons"]["0:0"]
    restored = ExperimentService(tmp_path, FakeTextRuntime())
    assert restored.snapshot(record["id"]) == record
    assert restored.history()[0]["kind"] == "text-generation"


def test_text_failure_restores_model(tmp_path):
    runtime = FakeTextRuntime(fail=True)
    service = ExperimentService(tmp_path, runtime)
    record = finish(service, service.start(text_request(), "text-generation"))
    assert record["status"] == "failed"
    assert runtime.restored and not runtime.pruning


def test_editable_text_api_is_retired_and_still_guarded(tmp_path):
    service = ExperimentService(tmp_path, FakeTextRuntime())
    client = TestClient(create_app(service))
    payload = text_request().model_dump()
    assert client.post("/api/text-runs", json=payload).status_code == 403
    headers = {"X-Lab-Token": client.get("/api/config").json()["token"]}
    response = client.post("/api/text-runs", headers=headers, json=payload)
    assert response.status_code == 410
    assert "fixed GPT-2 walkthrough" in response.json()["detail"]
    assert service.active is None


def test_text_request_bounds():
    with pytest.raises(ValidationError, match="at most 11"):
        TextRunRequest(prompt="Data", variants=[{"technique":"layers","strength":12}])
    with pytest.raises(ValidationError):
        TextRunRequest(prompt="x" * 1501, variants=[{"technique":"heads","strength":1}])


def test_text_cancellation_keeps_partial_results(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class BlockingRuntime(FakeTextRuntime):
        def text_answer(self, prompt, cancel, topology):
            if self.pruning:
                entered.set()
                release.wait(5)
                return {"text":"unfinished", "pipeline":{}}
            return super().text_answer(prompt, cancel, topology)
    runtime = BlockingRuntime()
    service = ExperimentService(tmp_path, runtime)
    run_id = service.start(text_request(), "text-generation")
    assert entered.wait(3)
    service.cancel(run_id)
    release.set()
    record = finish(service, run_id)
    assert record["status"] == "cancelled"
    assert runtime.restored and not runtime.pruning
    assert "0" in record["baselines"] and not record["comparisons"]
