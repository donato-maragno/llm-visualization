"""Question-linked diagrams must use actual generation inputs and actual masks."""

import json
import threading

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM

from src.coding_problems import Problem
from src.judge_eval import prepare_generation_inputs
from src.question_pipeline import capture_question_pipeline, MAX_DISPLAY_TOKENS
from src.web_service import ExperimentService, Runtime, RunRequest, Variant
from web_app import create_app


class Tokenizer:
    eos_token_id = 31

    def __init__(self, chat=False):
        self.chat_template = "test-chat" if chat else None

    def apply_chat_template(self, messages, **kwargs):
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, truncation=False, max_length=None, **kwargs):
        values = [1 + ord(char) % 29 for char in text]
        if truncation and max_length is not None:
            values = values[:max_length]
        ids = torch.tensor([values])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, **kwargs):
        return "".join(f"t{int(value)}" for value in ids)


def problem(id="double", title="Double a number"):
    return Problem(id=id, title=title, description="Implement solve(n). Return twice n.",
                   tests=[{"args":[3], "expected":6}, {"args":[0], "expected":0}])


@pytest.fixture(params=["gpt2", "qwen2"])
def runtime(request):
    torch.manual_seed(7)
    runtime = Runtime()
    if request.param == "gpt2":
        runtime.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=1024, n_embd=32, n_head=4, n_layer=2,
            attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
        )).eval()
    else:
        runtime.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, max_position_embeddings=1024, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        )).eval()
    runtime.tokenizer = Tokenizer(chat=request.param == "qwen2")
    return runtime


def test_question_snapshot_uses_full_generation_context(runtime):
    model, tokenizer = runtime.model, runtime.tokenizer
    implementation = model.config._attn_implementation
    question = problem()
    prepared, inputs = prepare_generation_inputs(model, tokenizer, question.prompt())
    view = capture_question_pipeline(model, tokenizer, question)
    json.dumps(view, allow_nan=False)
    assert view["problem_id"] == question.id
    assert view["prompt"] == prepared
    assert view["input_text"] == tokenizer.decode(inputs.input_ids[0].tolist())
    assert view["input_token_count"] == inputs.input_ids.shape[1] > 32
    assert len(view["tokens"]) == MAX_DISPLAY_TOKENS
    assert view["tokens"][-1]["position"] == inputs.input_ids.shape[1] - 1
    assert view["sampled_positions"]
    assert model.config._attn_implementation == implementation
    positions = [token["position"] for token in view["tokens"]]
    assert [token["id"] for token in view["tokens"]] == inputs.input_ids[0, positions].tolist()
    if model.config.model_type == "qwen2":
        assert view["position_encoding"] == "rotary"
        assert set(view["embeddings"]) == {"token"}, "Do not invent additive Qwen positional embeddings"
    else:
        assert view["position_encoding"] == "learned"
        assert set(view["embeddings"]) == {"token", "position", "combined"}
    try:
        model.set_attn_implementation("eager")
        with torch.inference_mode():
            expected = model(**inputs, output_attentions=True, use_cache=False)
        for index, layer in enumerate(view["layers"]):
            actual = torch.tensor(layer["attention"])
            reference = expected.attentions[index][0][:, positions][:, :, positions]
            assert torch.allclose(actual, reference, atol=2e-6)
            omitted = torch.tensor(layer["omitted_attention"])
            assert torch.allclose(actual.sum(-1) + omitted, torch.ones_like(omitted), atol=1e-5)
            assert torch.any(omitted[:, -1] > .1), "Unshown prompt keys must keep their probability mass"
            assert torch.count_nonzero(actual.triu(1)) == 0
        expected_probabilities = expected.logits[0, positions].float().softmax(-1)
        for i, prediction in enumerate(view["predictions"]):
            for candidate in prediction["candidates"]:
                assert candidate["probability"] == pytest.approx(expected_probabilities[i, candidate["id"]].item(), abs=1e-6)
    finally:
        model.set_attn_implementation(implementation)


@pytest.mark.parametrize("technique,strength", [("heads",1),("layers",1),("magnitude",30)])
def test_question_masks_are_measured_and_reverted(runtime, technique, strength):
    original = runtime.question_pipeline(problem(), threading.Event())
    before = {name:value.clone() for name,value in runtime.model.state_dict().items()}
    variant = Variant(technique=technique, strength=strength)
    with runtime.pruned(variant, 42):
        view = runtime.question_pipeline(problem(), threading.Event(), variant)
        assert view["predictions"] != original["predictions"]
        if technique == "heads":
            for layer in view["layers"]:
                assert layer["masked_heads"] == [0]
                assert not torch.count_nonzero(torch.tensor(layer["head_output"][0]))
                assert torch.count_nonzero(torch.tensor(layer["attention"][0]))
        elif technique == "layers":
            assert view["layers"][-1]["bypassed"]
            assert view["layers"][-1]["input"] == view["layers"][-1]["output"]
        else:
            assert all(layer["masked_weights"] > 0 for layer in view["layers"])
    assert runtime._active_pruner is None
    for name, value in runtime.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_question_capture_failure_cleans_up_hooks(runtime, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Snapshot forward failed")
    monkeypatch.setattr(runtime.model, "forward", fail)
    with pytest.raises(RuntimeError, match="Snapshot forward failed"):
        runtime.question_pipeline(problem(), threading.Event())
    for module in runtime.model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


class QuestionRuntime:
    def load(self, model_name):
        pass
    def answer(self, question, tokens, cancel):
        return {"text":"def solve(n): return n * 2", "seconds":.1, "tokens":12}
    def question_pipeline(self, question, cancel, variant=None):
        return {"schema":"question-forward-v1", "problem_id":question.id, "prompt":question.prompt(),
                "variant":variant.model_dump() if variant else None, "large_tensor_sample":[1,2,3]}
    def pruned(self, variant, seed):
        from contextlib import nullcontext
        return nullcontext()


def finish(service, request):
    run_id = service.start(request)
    service.worker.join(timeout=10)
    record = service.snapshot(run_id)
    assert record["status"] == "completed", record["error"]
    return record


def test_each_tested_question_has_its_own_paired_snapshots(tmp_path):
    request = RunRequest(problems=[problem(), problem("other-double", "Another double")],
                         variants=[{"technique":"heads", "strength":1}], execution_consent=True)
    service = ExperimentService(tmp_path, QuestionRuntime())
    record = finish(service, request)
    assert record["evaluation"] == "python-outcomes-v1"
    for qi, question in enumerate(request.problems):
        baseline = record["baselines"][str(qi)]
        cell = record["comparisons"][f"0:{qi}"]
        assert cell["score"] == 1
        assert baseline["execution"]["passed"] == cell["answer"]["execution"]["passed"] == 2
        pair = service.pipeline_pair(record["id"], qi, 0)
        assert pair["baseline"]["problem_id"] == pair["pruned"]["problem_id"] == question.id
        assert pair["baseline"]["prompt"] == pair["pruned"]["prompt"] == question.prompt()
        assert pair["pruned"]["variant"] == request.variants[0].model_dump()
    restored = ExperimentService(tmp_path, QuestionRuntime())
    assert restored.pipeline_pair(record["id"], 1, 0) == service.pipeline_pair(record["id"], 1, 0)


def test_polling_excludes_tensors_but_export_and_selected_pair_include_them(tmp_path):
    request = RunRequest(problems=[problem()], variants=[{"technique":"heads", "strength":1}], execution_consent=True)
    service = ExperimentService(tmp_path, QuestionRuntime())
    record = finish(service, request)
    client = TestClient(create_app(service))
    url = f"/api/runs/{record['id']}"
    compact = client.get(url).json()
    assert compact["baselines"]["0"]["pipeline_available"]
    assert "pipeline" not in compact["baselines"]["0"]
    assert "pipeline" not in compact["comparisons"]["0:0"]["answer"]
    assert client.get(url + "?include_pipelines=true").json() == record
    pair = client.get(url + "/pipeline?question=0&variant=0").json()
    assert pair["baseline"]["problem_id"] == pair["pruned"]["problem_id"] == "double"
    assert client.get(url + "/pipeline?question=-1").status_code == 422
    assert client.get(url + "/pipeline?question=1").status_code == 422
    assert client.get(url + "/pipeline?variant=2").status_code == 422
    service.runs[record["id"]]["status"] = "running"
    assert client.get(url + "/pipeline").status_code == 409


def test_visualization_failure_does_not_change_solution_scores(tmp_path):
    class FailedCapture(QuestionRuntime):
        def question_pipeline(self, *args, **kwargs):
            raise RuntimeError("Measurement unavailable")
    request = RunRequest(problems=[problem()], variants=[{"technique":"heads", "strength":1}], execution_consent=True)
    record = finish(ExperimentService(tmp_path, FailedCapture()), request)
    assert record["comparisons"]["0:0"]["score"] == 1
    assert "pipeline_error" in record["baselines"]["0"]
    assert record["baselines"]["0"]["execution"]["passed"] == 2
