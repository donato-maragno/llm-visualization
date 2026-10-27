"""Coding-only web lifecycle tests: no judge, network, or model downloads."""

import json
import threading
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.coding_problems import CATALOG
from src.model_loader import DEFAULT_MODEL_NAME
from src.web_service import ExperimentService, RunRequest, Variant
from web_app import create_app


class FakeRuntime:
    def __init__(self, fail=False):
        self.pruning = False
        self.fail = fail
        self.restored = False
        self.loads = []

    def load(self, model_name=None):
        self.loads.append(model_name)

    def answer(self, question, tokens, cancel):
        assert tokens is None, "The coding app must not impose a generation token cap"
        assert "Python" in question and "solve" in question
        if self.fail and self.pruning:
            raise ValueError("Inference failed")
        code = "def solve(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target-n in seen: return [seen[target-n], i]\n        seen[n] = i"
        return {"text": code, "seconds": 0.2, "tokens": 50}

    @contextmanager
    def pruned(self, variant, seed):
        self.pruning = True
        try:
            yield
        finally:
            self.pruning = False
            self.restored = True


def finish(service, run_id):
    service.worker.join(timeout=15)
    record = service.snapshot(run_id)
    assert record["status"] not in ("queued", "running", "cancelling")
    return record


def request(**overrides):
    return RunRequest(**{
        "problems": [CATALOG[0].model_dump()],
        "variants": [{"technique": "magnitude", "strength": 20}],
        "execution_consent": True,
        **overrides,
    })


def test_live_experiment_and_persistence(tmp_path):
    runtime = FakeRuntime()
    service = ExperimentService(tmp_path, runtime)
    run_id = service.start(request())
    record = finish(service, run_id)
    assert record["status"] == "completed", record["error"]
    assert record["done"] == record["total"] == 2
    assert record["comparisons"]["0:0"]["score"] == 1
    assert record["baselines"]["0"]["execution"]["passed"] == 4
    assert runtime.restored
    assert record["evaluation"] == "python-outcomes-v1"
    assert ExperimentService(tmp_path, FakeRuntime()).snapshot(run_id) == record


def test_failure_restores_model(tmp_path):
    runtime = FakeRuntime(fail=True)
    service = ExperimentService(tmp_path, runtime)
    record = finish(service, service.start(request()))
    assert record["status"] == "failed"
    assert runtime.restored and not runtime.pruning


def test_cancel_and_single_worker(tmp_path):
    release, entered = threading.Event(), threading.Event()
    class BlockingRuntime(FakeRuntime):
        def load(self, model_name=None):
            entered.set()
            release.wait(3)
    service = ExperimentService(tmp_path, BlockingRuntime())
    run_id = service.start(request())
    assert entered.wait(2)
    with pytest.raises(RuntimeError):
        service.start(request())
    service.cancel(run_id)
    release.set()
    assert finish(service, run_id)["status"] == "cancelled"


def test_api_validation_and_security(tmp_path):
    service = ExperimentService(tmp_path, FakeRuntime())
    client = TestClient(create_app(service))
    assert client.get("/").status_code == 200
    config = client.get("/api/config").json()
    assert "api_key" not in config and "judge_model" not in config
    assert "Coder" in config["model"]
    assert len(config["problems"]) == 10
    assert client.post("/api/runs", json={}).status_code == 403
    headers = {"X-Lab-Token": config["token"]}
    assert client.post("/api/runs", headers=headers, json={}).status_code == 422
    assert client.get("/api/runs/not-found").status_code == 404
    response = client.post("/api/runs", headers=headers, json=request().model_dump())
    assert response.status_code == 202
    assert "Content-Security-Policy" in response.headers
    finish(service, response.json()["id"])
    added = client.post("/api/problems/new", headers=headers, json={"existing":[p["id"] for p in config["problems"]]})
    assert added.status_code == 200 and added.json()["tests"]


@pytest.mark.parametrize("technique,strength", [("layers",28),("heads",12),("random",81),("magnitude",0)])
def test_variant_bounds(technique, strength):
    with pytest.raises(ValidationError):
        Variant(technique=technique, strength=strength)


def test_execution_consent_is_required():
    with pytest.raises(ValidationError, match="Confirm local"):
        request(execution_consent=False)


def test_judge_and_token_cap_settings_are_removed():
    with pytest.raises(ValidationError):
        request(judge="openrouter")
    with pytest.raises(ValidationError):
        request(max_new_tokens=128)


def test_interrupted_history_and_legacy_files(tmp_path):
    service = ExperimentService(tmp_path, FakeRuntime())
    run_id = service.start(request())
    finish(service, run_id)
    path = tmp_path / f"{run_id}.json"
    data = json.loads(path.read_text())
    data["status"] = "running"
    path.write_text(json.dumps(data))
    assert ExperimentService(tmp_path, FakeRuntime()).snapshot(run_id)["status"] == "interrupted"
    del data["evaluation"]
    path.write_text(json.dumps(data))
    assert ExperimentService(tmp_path, FakeRuntime()).history() == []


def test_gpt2_does_not_load_a_judge(tmp_path):
    runtime = FakeRuntime()
    service = ExperimentService(tmp_path, runtime)
    record = finish(service, service.start(request(model_name="gpt2")))
    assert record["model"] == "gpt2" and record["phase"] == "complete"
    assert record["mask_active"] is False
    assert all(model == "gpt2" for model in runtime.loads)


@pytest.mark.parametrize("model_name", ["gpt2", DEFAULT_MODEL_NAME])
@pytest.mark.parametrize("truncated", [False, True])
def test_invalid_generated_answers_are_diagnosed_and_score_zero(tmp_path, model_name, truncated):
    class InvalidRuntime(FakeRuntime):
        def answer(self, question, tokens, cancel):
            answer = super().answer(question, tokens, cancel)
            if not truncated:
                answer["text"] = question + "\n\nAnswer: " + question
            answer["truncated"] = truncated
            return answer

    runtime = InvalidRuntime()
    service = ExperimentService(tmp_path, runtime)
    record = finish(service, service.start(request(model_name=model_name)))
    assert record["status"] == "completed"
    assert record["done"] == record["total"] == 2
    assert runtime.restored and not runtime.pruning
    assert record["comparisons"]["0:0"]["score"] == 0
    for answer in (record["baselines"]["0"], record["comparisons"]["0:0"]["answer"]):
        execution = answer["execution"]
        assert execution["status"] == "invalid"
        assert execution["passed"] == 0 and execution["cases"] == []
        assert "<unknown>" not in execution["error"]
        assert ("context limit" if truncated else "not valid Python") in execution["error"]
        if model_name == "gpt2":
            assert "Select Qwen2.5-Coder" in execution["error"]
    assert ExperimentService(tmp_path, FakeRuntime()).snapshot(record["id"]) == record
