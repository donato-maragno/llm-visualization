"""
Model downloading, loading, and caching.

Handles the initial download of a suitable small LLM and provides
helpers to load the model and tokenizer for experiments.

Default model: Qwen2.5-Coder-1.5B-Instruct
  - 28 layers, 12 attention heads (GQA: 12 Q / 2 KV)
  - 1.54B parameters (~3 GB in FP16)
  - Apache 2.0 license — fully open and practical on an 8 GB laptop GPU
  - Substantially stronger than GPT-2 (124M) for evaluating pruning effects
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .utils import get_device


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[2] / "models")


def list_available_models() -> list[str]:
    """Return recommended models for pruning experiments, from lightest to strongest."""
    return [
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",    # 0.5B — fastest, smallest
        DEFAULT_MODEL_NAME,                # 1.5B — recommended for 8 GB GPUs
        "Qwen/Qwen2.5-Coder-3B-Instruct",      # 3B — tighter VRAM headroom; check its license
    ]


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Download (if needed) and load a HuggingFace causal LM and tokenizer.

    The model is placed in evaluation mode and moved to the selected device.
    CUDA loads use FP16 to reduce VRAM usage; CPU loads use FP32.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    if device is None:
        device = get_device()

    print(f"Loading model '{model_name}' on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    # FP16 cuts GPU weight memory roughly in half versus FP32, which makes
    # the 1.5B default comfortable on an 8 GB RTX 4070 Laptop GPU. Keep CPU
    # loading in FP32 for broad compatibility.
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=model_dtype,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if device.type != "cpu":
        model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({model_dtype})")
    print(f"  Done.\n")
    return model, tokenizer


def get_model_metadata(model: PreTrainedModel) -> dict:
    """Extract architecture metadata useful for pruning configuration."""
    config = model.config
    info = {"model_type": config.model_type}

    for attr, key in [
        ("num_hidden_layers", "num_layers"),
        ("n_layer", "num_layers"),
        ("num_attention_heads", "num_heads"),
        ("n_head", "num_heads"),
        ("hidden_size", "hidden_size"),
        ("n_embd", "hidden_size"),
    ]:
        if hasattr(config, attr):
            info[key] = getattr(config, attr)

    # GQA: number of KV heads (Qwen2, Llama3)
    if hasattr(config, "num_key_value_heads"):
        info["num_kv_heads"] = config.num_key_value_heads

    return info


def save_backup(model: PreTrainedModel, backup_path: str) -> None:
    """Save the current model weights as a backup checkpoint."""
    os.makedirs(os.path.dirname(backup_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), backup_path)
    print(f"Backup saved to {backup_path}")


def restore_original_weights(model: PreTrainedModel, backup_path: str) -> None:
    """Restore a model's state from a previously saved checkpoint."""
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    state = torch.load(backup_path, map_location=get_device())
    model.load_state_dict(state)
    print(f"Restored original weights from {backup_path}")
