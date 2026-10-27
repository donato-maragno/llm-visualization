"""
Structural pruning utilities.

Structural (or "hard") pruning physically modifies or removes components
from the model.  Unlike logical masking, structural pruning **changes** the
model graph, reduces the parameter count, and can yield real inference
speed-ups and memory savings.

Advantages
----------
- **Faster inference** -- fewer FLOPs and less memory bandwidth.
- **Lower memory** -- pruned parameters are freed.
- **Permanent** -- guarantees pruned components can never contribute.

Limitations
-----------
- **Irreversible without a backup** -- the original weights are gone.
  Always save a checkpoint before applying structural pruning.
- **Architecture-dependent** -- removing components may break assumptions
  made by the model code (e.g. head count, hidden size).
- **Residual connections** -- layers with skip connections cannot simply
  be removed without adjusting the residual path.  This module handles
  the common cases but not every edge case.

When to use
-----------
Use structural pruning **after** logical masking has helped you identify
which components are safe to remove.  Apply it for deployment or for
benchmarking real-world performance gains.
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import count_parameters


# ---------------------------------------------------------------------------
# Module replacement helpers
# ---------------------------------------------------------------------------

class ZeroPlaceholder(nn.Module):
    """A module that outputs zeros of a pre-recorded shape.

    Unlike ``nn.Identity``, this truly zeros the contribution of the
    original module rather than passing input through unchanged.

    Usage::

        placeholder = ZeroPlaceholder.from_module(original_module, sample_input)
        # Replace original_module with placeholder in the parent.
    """

    def __init__(self, output_shape: torch.Size) -> None:
        super().__init__()
        self.output_shape = output_shape

    def forward(self, *args, **kwargs):
        return torch.zeros(self.output_shape, device=next(self.parameters()).device
                           if list(self.parameters()) else torch.device("cpu"))

    @classmethod
    def from_module(cls, module: nn.Module, sample_input, **kwargs):
        """Infer output shape by running a sample forward pass."""
        with torch.no_grad():
            out = module(sample_input, **kwargs)
        shape = out.shape if isinstance(out, torch.Tensor) else out[0].shape
        return cls(shape)


def replace_module(
    parent: nn.Module,
    child_name: str,
    replacement: nn.Module,
) -> nn.Module:
    """Replace a child module inside a parent module.

    Args:
        parent: The containing ``nn.Module``.
        child_name: Attribute name of the child to replace.
        replacement: The new module to put in its place.

    Returns:
        The **original** module that was removed (so the caller can
        restore it later if desired).
    """
    original = getattr(parent, child_name)
    setattr(parent, child_name, replacement)
    return original


# ---------------------------------------------------------------------------
# Layer removal (for residual / sequential architectures)
# ---------------------------------------------------------------------------

def remove_transformer_layers(
    model: nn.Module,
    layer_indices: List[int],
    mode: str = "identity",
) -> Dict[int, nn.Module]:
    """Structurally remove transformer layers from a GPT-2 style model.

    Replaces each pruned layer with either an ``nn.Identity`` or a
    ``ZeroPlaceholder``.  Because GPT-2 uses residual connections
    (``x = x + layer(x)``), an identity replacement is equivalent to
    the layer contributing zero -- the residual path carries the input
    through unchanged.

    Args:
        model: A GPT-2 / OPT style model whose layers live in
            ``model.transformer.h`` or ``model.model.decoder.layers``.
        layer_indices: 0-based indices of layers to prune.
        mode: ``"identity"`` (default) or ``"zero"``.

    Returns:
        Dict mapping layer index → original module, so layers can be
        restored later.
    """
    # Find the layer list
    if hasattr(model, "transformer"):
        layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers
    else:
        raise RuntimeError("Cannot locate transformer layers in model.")

    removed = {}
    for idx in sorted(layer_indices, reverse=True):
        if mode == "zero":
            replacement = _ZeroIdentity()
        else:
            replacement = nn.Identity()
        removed[idx] = layers[idx]
        layers[idx] = replacement

    after = count_parameters(model)
    return removed


class _ZeroIdentity(nn.Module):
    """Acts like identity for shapes but outputs zeros (helper for layer removal)."""
    def forward(self, *args, **kwargs):
        # GPT-2 block forward returns (hidden_states, present_key_value?)
        # but when removed we only care about hidden_states.
        x = args[0]
        return (torch.zeros_like(x),)


# ---------------------------------------------------------------------------
# Head pruning (structural -- zeros QKV weight slices)
# ---------------------------------------------------------------------------

def structurally_prune_heads(
    model: nn.Module,
    per_layer_heads: Dict[int, List[int]],
    head_dim: int,
    num_heads: int,
) -> None:
    """Zero the QKV (and optionally output) weight slices for pruned heads.

    This is the structural counterpart to ``masking.mask_attention_heads``.
    It permanently zeros the relevant weight slices so those heads never
    contribute.  Weight shapes are unchanged.

    Args:
        model: GPT-2 style model.
        per_layer_heads: ``{layer_idx: [head_idx, …]}`` mapping.
        head_dim: Dimensionality of a single head.
        num_heads: Total number of heads per layer.
    """
    if hasattr(model, "transformer"):
        layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers
    else:
        raise RuntimeError("Cannot locate transformer layers.")

    hidden = head_dim * num_heads

    for layer_idx, heads in per_layer_heads.items():
        attn = layers[layer_idx].attn if hasattr(layers[layer_idx], "attn") else layers[layer_idx].self_attn

        # Fused QKV (GPT-2)
        if hasattr(attn, "c_attn"):
            w = attn.c_attn.weight.data
            b = attn.c_attn.bias.data if attn.c_attn.bias is not None else None
            for h in heads:
                s = h * head_dim
                e = s + head_dim
                for offset in [0, hidden, 2 * hidden]:  # Q, K, V
                    w[s + offset : e + offset, :] = 0.0
                    if b is not None:
                        b[s + offset : e + offset] = 0.0
            continue

        # Separate Q, K, V (OPT)
        for proj in ("q_proj", "k_proj", "v_proj"):
            proj_mod = getattr(attn, proj, None)
            if proj_mod is None:
                continue
            for h in heads:
                s = h * head_dim
                e = s + head_dim
                proj_mod.weight.data[s:e, :] = 0.0
                if proj_mod.bias is not None:
                    proj_mod.bias.data[s:e] = 0.0


# ---------------------------------------------------------------------------
# Weight pruning helpers
# ---------------------------------------------------------------------------

def sparsify_tensor(
    tensor: torch.Tensor,
    sparsity: float,
    strategy: str = "magnitude",
) -> torch.Tensor:
    """Zero out a fraction of entries in a tensor in-place.

    Args:
        tensor: Tensor to sparsify (modified in-place).
        sparsity: Fraction of entries to zero (0-1).
        strategy: ``"magnitude"`` or ``"random"``.

    Returns:
        The same Tensor (modified in-place), for chaining.
    """
    if sparsity <= 0:
        return tensor

    if strategy == "magnitude":
        flat = tensor.abs().view(-1)
        k = max(1, int(flat.numel() * (1 - sparsity)))
        threshold = torch.topk(flat, k).values[-1]
        tensor[tensor.abs() < threshold] = 0.0
    elif strategy == "random":
        mask = torch.rand_like(tensor) > sparsity
        tensor *= mask
    else:
        raise ValueError(f"Unknown sparsification strategy: {strategy!r}")

    return tensor
