"""
Logical masking utilities.

Logical (or "soft") masking prevents selected components from contributing
to the computation **without** modifying the model architecture.  The model
graph, parameter count, and output shapes stay identical to the original.

Advantages
----------
- **Reversible** -- remove the mask and the model is back to normal.
- **Safe** -- no risk of breaking downstream code that expects a specific
  output shape or parameter layout.
- **Fast to toggle** -- ideal for A / B comparisons during experimentation.

Limitations
-----------
- **No speed-up** -- masked components still consume compute (they just
  produce zeros).  Inference latency may actually *increase* slightly
  because of the extra masking operations.
- **No memory savings** -- all parameters remain in GPU / CPU memory.

When to use
-----------
Use logical masking during the **exploration phase** of your pruning
research: quickly try many configurations, measure output quality impact,
and identify which components are safe to prune before committing to
structural changes.
"""

from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Generic hook helpers
# ---------------------------------------------------------------------------

def register_zero_hook(module: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that replaces the module's output with zeros.

    The hook preserves the shape of the original output (including tuple
    outputs, as produced by some attention modules with caching).

    Args:
        module: The module to hook.

    Returns:
        A handle that can be used to remove the hook via ``handle.remove()``.
    """
    def _zero(mod, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)

    return module.register_forward_hook(_zero)


def register_scale_hook(
    module: nn.Module, factor: float = 0.5
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that scales the module's output by `factor`.

    Args:
        module: The module to hook.
        factor: Multiplier applied to the output (0 = full mask, 1 = no effect).

    Returns:
        A removable hook handle.
    """
    def _scale(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] * factor,) + out[1:]
        return out * factor

    return module.register_forward_hook(_scale)


# ---------------------------------------------------------------------------
# Head-level masking (standalone -- works with GPT-2 / OPT attention)
# ---------------------------------------------------------------------------

def mask_attention_heads(
    attn_module: nn.Module,
    head_indices: List[int],
    num_heads: int,
    head_dim: int,
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that zeros selected attention heads.

    Args:
        attn_module: The attention ``nn.Module`` (e.g. ``GPT2Attention``).
        head_indices: List of head indices to zero (0-based).
        num_heads: Total number of attention heads in this module.
        head_dim: Dimensionality of each head.

    Returns:
        A removable hook handle.
    """
    from .pruners.head_pruner import HeadPruner
    helper = HeadPruner(attn_module)
    projection = helper._get_output_projection(attn_module)
    return projection.register_forward_pre_hook(
        helper._make_head_prehook(head_indices, num_heads, head_dim)
    )


# ---------------------------------------------------------------------------
# Weight-level masking
# ---------------------------------------------------------------------------

def create_weight_mask(
    param: nn.Parameter,
    sparsity: float = 0.5,
    strategy: str = "magnitude",
    threshold: float = 0.01,
) -> torch.Tensor:
    """Create a binary mask for a single parameter tensor.

    Does **not** register hooks or modify the model.  Returns the mask
    so the caller can decide how to apply it.

    Args:
        param: The parameter tensor to mask.
        sparsity: Fraction of weights to prune (for magnitude / random).
        strategy: ``"magnitude"``, ``"random"``, or ``"threshold"``.
        threshold: Absolute-value cutoff for ``"threshold"`` strategy.

    Returns:
        A binary float tensor (same shape as param, values 0 or 1).
    """
    data = param.data
    shape = data.shape
    device = data.device

    if strategy == "magnitude":
        from .pruners.weight_pruner import WeightPruner
        return WeightPruner._magnitude_mask(param, sparsity)

    if strategy == "random":
        return (torch.rand(shape, device=device) > sparsity).float()

    if strategy == "threshold":
        return (data.abs() >= threshold).float()

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Convenience: apply masking to a list of modules by name
# ---------------------------------------------------------------------------

def apply_mask_to_modules(
    model: nn.Module,
    module_names: List[str],
    mode: str = "zero",
) -> List[torch.utils.hooks.RemovableHandle]:
    """Apply logical masking to a list of named modules.

    Args:
        model: The model containing the modules.
        module_names: Dotted module names to mask (e.g.
            ``["transformer.h.0", "transformer.h.11"]``).
        mode: ``"zero"`` (default) or a float string for scaling.

    Returns:
        List of removable hook handles.  Call ``h.remove()`` on each to undo.
    """
    handles = []
    for name in module_names:
        mod = model
        for part in name.split("."):
            if part.isdigit():
                part = int(part)
            if isinstance(part, int) and isinstance(mod, nn.ModuleList):
                mod = mod[part]
            else:
                mod = getattr(mod, part)
        if mode == "zero":
            handles.append(register_zero_hook(mod))
        else:
            factor = float(mode)
            handles.append(register_scale_hook(mod, factor))
    return handles
