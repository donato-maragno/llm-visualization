"""
Basic smoke tests for the pruning framework.

Run with:
    pytest tests/ -v
    or
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from src.utils import count_parameters, count_nonzero_parameters, model_size_mb
from src.model_loader import DEFAULT_MODEL_NAME
from transformers import Qwen2Config, Qwen2ForCausalLM
from src.pruners.base_pruner import BasePruner
from src.pruners.module_pruner import ModulePruner
from src.pruners.head_pruner import HeadPruner
from src.pruners.weight_pruner import WeightPruner


@pytest.fixture(scope="module")
def model_tokenizer():
    """Build a tiny local Qwen2 model so tests need no model download."""
    config = Qwen2Config(
        vocab_size=503,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config), None


class TestUtils:
    def test_count_params_simple(self):
        model = torch.nn.Linear(10, 5)
        assert count_parameters(model) == 55

    def test_model_size_mb(self):
        model = torch.nn.Linear(100, 200)
        size = model_size_mb(model)
        assert 0.01 < size < 1.0


class TestModelArchitecture:
    def test_build_qwen2_model(self, model_tokenizer):
        model, _ = model_tokenizer
        assert model is not None

    def test_default_model_is_open_qwen(self):
        assert DEFAULT_MODEL_NAME == "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    def test_param_count(self, model_tokenizer):
        model, _ = model_tokenizer
        n = count_parameters(model)
        assert 0 < n < 100_000


class _ConcreteTestPruner(BasePruner):
    def apply_masking(self):
        pass

    def apply_structural(self):
        pass


class TestBasePruner:
    def test_resolve_int(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = _ConcreteTestPruner(model)
        assert pruner._resolve_indices(3, 12) == [3]

    def test_resolve_all(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = _ConcreteTestPruner(model)
        assert pruner._resolve_indices("all", 12) == list(range(12))

    def test_resolve_even_odd(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = _ConcreteTestPruner(model)
        assert pruner._resolve_indices("even", 12) == [0, 2, 4, 6, 8, 10]
        assert pruner._resolve_indices("odd", 12) == [1, 3, 5, 7, 9, 11]

    def test_resolve_slice(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = _ConcreteTestPruner(model)
        assert pruner._resolve_indices("0:6", 12) == [0, 1, 2, 3, 4, 5]

    def test_resolve_out_of_range(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = _ConcreteTestPruner(model)
        with pytest.raises(ValueError):
            pruner._resolve_indices([0, 100], 12)

    def test_snapshot_and_revert(self, model_tokenizer):
        model, _ = model_tokenizer
        original = model.state_dict()["model.embed_tokens.weight"].clone()
        pruner = _ConcreteTestPruner(model)
        pruner.snapshot()
        model.model.embed_tokens.weight.data.zero_()
        pruner.revert()
        assert torch.allclose(model.state_dict()["model.embed_tokens.weight"], original)


class TestModulePruner:
    def test_masking(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = ModulePruner(model, {
            "target_modules": ["model.layers.1"],
            "mode": "masking",
        })
        pruner.snapshot()
        pruner.apply_masking()
        inp = torch.randint(0, model.config.vocab_size, (1, 5))
        with torch.no_grad():
            out = model(inp)
        assert out.logits.shape == (1, 5, model.config.vocab_size)
        pruner.revert()

    def test_structural(self, model_tokenizer):
        model, _ = model_tokenizer
        before = count_parameters(model)
        pruner = ModulePruner(model, {
            "target_modules": ["model.layers.1"],
            "mode": "structural",
            "structural_replacement": "identity",
        })
        pruner.snapshot()
        pruner.apply_structural()
        after = count_parameters(model)
        assert after < before
        pruner.revert()


class TestHeadPruner:
    def test_masking(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = HeadPruner(model, {
            "head_indices": [0, 1],
            "mode": "masking",
        })
        pruner.snapshot()
        pruner.apply_masking()
        inp = torch.randint(0, model.config.vocab_size, (1, 5))
        with torch.no_grad():
            out = model(inp)
        assert out is not None
        pruner.revert()

    def test_per_layer(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = HeadPruner(model, {
            "head_indices": {0: [0], 1: [1, 2]},
            "mode": "masking",
        })
        pruner.snapshot()
        pruner.apply_masking()
        pruner.revert()


class TestWeightPruner:
    def test_magnitude_masking(self, model_tokenizer):
        model, _ = model_tokenizer
        pruner = WeightPruner(model, {
            "strategy": "magnitude",
            "sparsity": 0.5,
            "target_modules": "all",
            "mode": "masking",
        })
        pruner.snapshot()
        pruner.apply_masking()
        inp = torch.randint(0, model.config.vocab_size, (1, 5))
        with torch.no_grad():
            out = model(inp)
        assert out is not None
        pruner.revert()

    def test_random_structural(self, model_tokenizer):
        model, _ = model_tokenizer
        before = count_nonzero_parameters(model)
        pruner = WeightPruner(model, {
            "strategy": "random",
            "sparsity": 0.3,
            "target_modules": "model.layers.0.mlp.*",
            "mode": "structural",
        })
        pruner.snapshot()
        pruner.apply_structural()
        after = count_nonzero_parameters(model)
        assert after < before
        pruner.revert()
