"""Separate educational GPT-2 page, locked input, fixed masks and Qwen-only lab."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import BatchEncoding

from src.coding_problems import CATALOG
from src.judge_eval import get_model_answer, prepare_generation_inputs
from src.model_loader import DEFAULT_MODEL_NAME
from src.web_service import ExperimentService
from src.walkthrough import WALKTHROUGH_VERSION, WALKTHROUGH_SENTENCE, WALKTHROUGH_PRESETS, WALKTHROUGH_SEED
from web_app import create_app


class DemoRuntime:
    def __init__(self):
        self.prompts=[]
        self.pruning=False
        self.restored=0
        self.loads=[]
    def load(self, model_name):
        self.loads.append(model_name)
    def text_answer(self, prompt, cancel, topology):
        assert not cancel.is_set()
        self.prompts.append(prompt)
        return {"text":" a whole number.","seconds":.1,"tokens":5,"stop_reason":"eos","truncated":False,
                "pipeline":{"prompt":prompt,"measured":True}}
    @contextmanager
    def pruned(self, variant, seed):
        assert seed==WALKTHROUGH_SEED
        self.pruning=True
        try:
            yield {"technique":variant.technique}
        finally:
            self.pruning=False
            self.restored+=1


@pytest.fixture
def lab(tmp_path):
    runtime=DemoRuntime()
    service=ExperimentService(tmp_path,runtime)
    client=TestClient(create_app(service))
    config=client.get("/api/walkthrough/config").json()
    return runtime,service,client,{"X-Lab-Token":config["token"]}


def test_pages_have_clear_navigation_and_no_walkthrough_prompt_editor(lab):
    _,_,client,_=lab
    home=client.get("/")
    assert home.status_code==200
    assert home.text==client.get("/walkthrough").text
    assert "GPT-2 walkthrough — LLM visualization" in home.text
    assert 'href="/experiments"' in home.text
    assert 'id="fixed-sentence"' in home.text
    assert WALKTHROUGH_SENTENCE in home.text
    assert "<textarea" not in home.text and "contenteditable" not in home.text
    for removed in ("generate-walkthrough", "stop-walkthrough", "walkthrough-history", "export-walkthrough",
                    "THE INPUT / FIXED FOR EVERY MODEL STATE", "This sentence is locked"):
        assert removed not in home.text
    experiments=client.get("/experiments")
    assert experiments.status_code==200
    assert "Qwen experiments — LLM visualization" in experiments.text
    assert 'value="gpt2"' not in experiments.text
    assert 'href="/"' in experiments.text


def test_config_distinguishes_models_budgets_and_presets(lab):
    _,_,client,_=lab
    config=client.get("/api/walkthrough/config").json()
    assert config["sentence"]==WALKTHROUGH_SENTENCE
    assert config["version"]==WALKTHROUGH_VERSION
    assert config["seed"]==42
    assert [p["technique"] for p in config["presets"]]==["layers","heads","magnitude","random"]
    assert all(p["explanation"] and p["inspect"] for p in config["presets"])
    assert config["context_window"]==1024 and config["generation_token_cap"] is None
    qwen=client.get("/api/config").json()
    assert [model["id"] for model in qwen["models"]]==[DEFAULT_MODEL_NAME]
    assert qwen["context_window"]==32768 and qwen["generation_token_cap"] is None


@pytest.mark.parametrize("endpoint", ["/api/walkthrough/runs", "/api/walkthrough/ready"])
@pytest.mark.parametrize("payload",[
    {"prompt":"A different sentence"},{"sentence":"Other"},{"model_name":"gpt2"},
    {"seed":9},{"variants":[{"technique":"heads","strength":10}]},
    {"max_new_tokens":100},{"strength":80},
])
def test_walkthrough_rejects_all_client_changes(lab,payload,endpoint):
    runtime,service,client,headers=lab
    response=client.post(endpoint,headers=headers,json=payload)
    assert response.status_code==422
    assert service.active is None and runtime.prompts==[]


def test_walkthrough_runs_same_locked_sentence_for_all_presets_without_python(lab,monkeypatch):
    runtime,service,client,headers=lab
    def forbidden(*args,**kwargs):
        pytest.fail("Sentence walkthrough must not run generated text as Python")
    monkeypatch.setattr("src.web_service.run_solution",forbidden)
    assert client.post("/api/walkthrough/runs",json={}).status_code==403
    response=client.post("/api/walkthrough/runs",headers=headers,json={})
    assert response.status_code==202
    service.worker.join(timeout=10)
    record=service.snapshot(response.json()["id"])
    assert record["status"]=="completed"
    assert record["kind"]=="walkthrough" and record["walkthrough_version"]==WALKTHROUGH_VERSION
    assert record["config"]["prompt"]==WALKTHROUGH_SENTENCE
    assert record["config"]["variants"]==[{key:p[key] for key in ("technique","strength")} for p in WALKTHROUGH_PRESETS]
    assert runtime.prompts==[WALKTHROUGH_SENTENCE]*5
    assert runtime.loads==["gpt2"]
    assert runtime.restored==4 and not runtime.pruning
    assert record["done"]==record["total"]==5
    assert "execution" not in record["baselines"]["0"]
    assert all("score" not in cell for cell in record["comparisons"].values())
    restored=ExperimentService(service.output_dir,DemoRuntime())
    assert restored.snapshot(record["id"])==record
    assert restored.history()[0]["walkthrough_version"]==WALKTHROUGH_VERSION


def test_qwen_api_refuses_gpt2_coding_requests(lab):
    runtime,service,client,headers=lab
    response=client.post("/api/runs",headers=headers,json={"model_name":"gpt2","problems":[CATALOG[0].model_dump()],
        "variants":[{"technique":"heads","strength":1}],"execution_consent":True})
    assert response.status_code==422
    assert "Qwen-Coder only" in response.json()["detail"]
    assert not runtime.loads and service.active is None


def prepare_example(lab):
    runtime,service,client,headers=lab
    response=client.post("/api/walkthrough/ready",headers=headers,json={})
    assert response.status_code==202
    service.worker.join(timeout=10)
    record=service.snapshot(response.json()["id"])
    assert record["status"]=="completed"
    return record


def test_ready_automatically_reuses_saved_results_without_generation(lab):
    runtime,service,client,headers=lab
    record=prepare_example(lab)
    assert client.post("/api/walkthrough/ready",json={}).status_code==403
    response=client.post("/api/walkthrough/ready",headers=headers,json={})
    assert response.json()["id"]==record["id"]
    assert len(service.runs)==1 and len(runtime.prompts)==5
    fresh_runtime=DemoRuntime()
    restored=ExperimentService(service.output_dir,fresh_runtime)
    fresh_client=TestClient(create_app(restored))
    token=fresh_client.get("/api/walkthrough/config").json()["token"]
    reused=fresh_client.post("/api/walkthrough/ready",headers={"X-Lab-Token":token},json={})
    assert reused.json()["id"]==record["id"]
    assert fresh_runtime.loads==[] and fresh_runtime.prompts==[]


def test_simultaneous_visits_share_one_preparation(tmp_path):
    entered,release=threading.Event(),threading.Event()
    class SlowRuntime(DemoRuntime):
        def load(self,model_name):
            super().load(model_name)
            entered.set()
            release.wait(5)
    runtime=SlowRuntime()
    service=ExperimentService(tmp_path,runtime)
    client=TestClient(create_app(service))
    headers={"X-Lab-Token":client.get("/api/walkthrough/config").json()["token"]}
    first=client.post("/api/walkthrough/ready",headers=headers,json={})
    try:
        assert entered.wait(3)
        second=client.post("/api/walkthrough/ready",headers=headers,json={})
        assert first.json()["id"]==second.json()["id"]
        assert len(service.runs)==1
    finally:
        release.set()
        service.worker.join(timeout=10)
    assert runtime.loads==["gpt2"] and len(runtime.prompts)==5


def test_ready_cache_is_available_even_when_qwen_is_busy(lab):
    _,service,client,headers=lab
    record=prepare_example(lab)
    service.runs["other-task"]={"kind":"experiment","created_at":record["created_at"]+1}
    service.active="other-task"
    try:
        response=client.post("/api/walkthrough/ready",headers=headers,json={})
        assert response.status_code==202 and response.json()["id"]==record["id"]
        assert service.active=="other-task"
    finally:
        service.active=None


def test_ready_waits_if_worker_busy_and_no_cache_exists(lab):
    runtime,service,client,headers=lab
    service.runs["other-task"]={"kind":"experiment","created_at":1}
    service.active="other-task"
    try:
        response=client.post("/api/walkthrough/ready",headers=headers,json={})
        assert response.status_code==409
        assert "automatically" in response.json()["detail"]
        assert runtime.loads==[] and service.active=="other-task"
    finally:
        service.active=None


@pytest.mark.parametrize("corruption", ["old_version", "missing_pipeline", "failed", "different_seed"])
def test_ready_does_not_reuse_outdated_or_incomplete_results(lab,corruption):
    runtime,service,client,headers=lab
    record=prepare_example(lab)
    saved=service.runs[record["id"]]
    if corruption=="old_version":saved["walkthrough_version"]="prime-number-v1"
    elif corruption=="missing_pipeline":del saved["comparisons"]["0:0"]["answer"]["pipeline"]
    elif corruption=="failed":saved["status"]="failed"
    else:saved["config"]["seed"]=123
    new=prepare_example(lab)
    assert new["id"]!=record["id"]
    assert len(runtime.prompts)==10 and len(service.runs)==2


def test_active_runs_are_scoped_to_the_correct_page(lab):
    _,service,client,_=lab
    service.runs["test-active"]={"kind":"walkthrough"}
    service.active="test-active"
    assert client.get("/api/walkthrough/config").json()["active_run"]=="test-active"
    qwen=client.get("/api/config").json()
    assert qwen["active_run"] is None and qwen["busy_elsewhere"]=="test-active"
    service.runs["test-active"]["kind"]="experiment"
    assert client.get("/api/config").json()["active_run"]=="test-active"
    demo=client.get("/api/walkthrough/config").json()
    assert demo["active_run"] is None and demo["busy_elsewhere"]=="test-active"
    service.active=None


def test_no_output_cap_uses_available_model_context_without_silent_input_truncation():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.zeros(1))
            self.config=SimpleNamespace(max_position_embeddings=32768)
            self.generation_config=SimpleNamespace(eos_token_id=9)
        def generate(self,**kwargs):
            assert kwargs["max_new_tokens"]==32765
            return torch.cat((kwargs["input_ids"],torch.tensor([[4,9]])),dim=-1)
    class Tokenizer:
        chat_template=None
        eos_token_id=9
        def __call__(self,prompt,**kwargs):
            assert kwargs["truncation"] is False
            return BatchEncoding({"input_ids":torch.tensor([[1,2,3]])})
        def decode(self,*args,**kwargs):
            return "answer"
    answer=get_model_answer(Model(),Tokenizer(),"Question",max_new_tokens=None,return_details=True)
    assert answer["text"]=="answer" and not answer["truncated"]
    assert answer["stop_reason"]=="eos"
    model=Model()
    model.config.max_position_embeddings=3
    with pytest.raises(ValueError,match="No input was silently truncated"):
        prepare_generation_inputs(model,Tokenizer(),"Question",max_new_tokens=None)
