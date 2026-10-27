"""GPT-2 visuals must reflect applied masks and never leak into Qwen runs."""

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel
from pydantic import ValidationError

from src.web_service import Runtime, RunRequest, Variant
from src.coding_problems import CATALOG


@pytest.fixture
def runtime():
    runtime = Runtime()
    runtime.model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=64, n_embd=32, n_head=4, n_layer=2,
        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
    )).eval()
    return runtime


@pytest.mark.parametrize("technique,strength", [("layers",1),("heads",1),("magnitude",20),("random",50)])
def test_gpt2_applied_masks_and_restoration(runtime, technique, strength):
    model = runtime.model
    before = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(inputs).logits.clone()
        with runtime.pruned(Variant(technique=technique, strength=strength), 42) as view:
            assert view["model"] == "gpt2"
            assert view["layer_count"] == 2
            assert len(view["layers"]) == 2
            assert not torch.equal(model(inputs).logits, expected)
            if technique == "layers":
                assert not view["layers"][0]["bypassed"]
                assert view["layers"][1]["bypassed"]
            elif technique == "heads":
                assert all(layer["masked_heads"] == [0] for layer in view["layers"])
            else:
                layer = view["layers"][0]
                assert layer["masked_weights"] > 0
                for weight in layer["mlp_weights"]:
                    param = dict(model.named_parameters())[weight["name"]]
                    assert sum(weight["sample"]) <= weight["masked"]
                    assert len(weight["sample"]) <= 96
                    assert weight["sample_values"] == param.detach().reshape(-1)[:96].float().tolist()
                    # Every sampled position described as masked is really zero.
                    for index, masked in enumerate(weight["sample"]):
                        if masked:
                            assert param.reshape(-1)[index] == 0
        assert torch.equal(model(inputs).logits, expected)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_gpt2_boundaries():
    with pytest.raises(ValidationError, match="GPT-2 has 12 layers"):
        RunRequest(model_name="gpt2", problems=[CATALOG[0]], execution_consent=True, variants=[{"technique":"layers", "strength":12}])


def test_model_name_is_allowlisted():
    with pytest.raises(ValidationError):
        RunRequest(model_name="untrusted/custom-code", problems=[CATALOG[0]], execution_consent=True, variants=[{"technique":"layers", "strength":1}])
