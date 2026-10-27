"""Regression tests for review findings, using tiny in-memory models."""

from types import SimpleNamespace
import threading

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from config.config_loader import create_pruner_from_config
from src.evaluation import compute_distribution_divergence, evaluate_perplexity
from src.masking import register_zero_hook
from src.pruners.module_pruner import ModulePruner
from src.web_service import Runtime, Variant


@pytest.fixture
def tiny():
    torch.manual_seed(1)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )).eval()


@pytest.mark.parametrize("technique,strength", [("layers",1),("heads",1),("magnitude",20),("random",20)])
def test_runtime_restores_even_when_generation_fails(tiny, technique, strength):
    runtime = Runtime()
    runtime.model = tiny
    before = {key: value.clone() for key, value in tiny.state_dict().items()}
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        original = tiny(inputs).logits.clone()
    with pytest.raises(RuntimeError):
        with runtime.pruned(Variant(technique=technique, strength=strength), 42) as visualization:
            assert visualization is None, "Only GPT-2 may expose a model visualization"
            with torch.no_grad():
                changed = tiny(inputs).logits
            assert not torch.equal(original, changed)
            raise RuntimeError("Generation failed")
    for key, tensor in tiny.state_dict().items():
        assert torch.equal(tensor, before[key])
    with torch.no_grad():
        assert torch.equal(tiny(inputs).logits, original)


def test_factory_imports(tiny):
    assert isinstance(create_pruner_from_config(tiny, {
        "type": "module", "mode": "masking", "target_modules": ["model.layers.1"],
    }), ModulePruner)


def test_zero_hook_keeps_optional_cache():
    class TupleModule(torch.nn.Module):
        def forward(self, value):
            return value + 1, None
    model = TupleModule()
    hook = register_zero_hook(model)
    output = model(torch.ones(1))
    assert output[0].item() == 0 and output[1] is None
    hook.remove()


class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.arange(10).reshape(1, -1))


def test_distribution_overlap_count_and_fp16_stability():
    class ExtremeModel(torch.nn.Module):
        def forward(self, inputs):
            logits = torch.full((*inputs.shape, 32), -1000.0, dtype=torch.float16)
            logits[..., 0] = 1000
            return SimpleNamespace(logits=logits)
    model = ExtremeModel()
    result = compute_distribution_divergence(model, model, Tokenizer(), ["text"],
                                             max_length=6, stride=3, device=torch.device("cpu"))
    assert result["total_tokens"] == 9
    assert result["kl_divergence"] == result["js_divergence"] == 0
    assert result["top1_agreement"] == 1


def test_perplexity_does_not_double_count_overlap():
    counts = []
    class UniformModel(torch.nn.Module):
        def forward(self, inputs, labels):
            counts.append(int((labels[:, 1:] != -100).sum()))
            return SimpleNamespace(loss=torch.tensor(2.0))
    value = evaluate_perplexity(UniformModel(), Tokenizer(), ["text"],
                               max_length=6, stride=3, device=torch.device("cpu"))
    assert sum(counts) == 9
    assert value == pytest.approx(7.389, abs=.001)
