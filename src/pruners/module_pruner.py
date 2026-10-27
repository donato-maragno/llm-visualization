"""
Module / layer pruning.

Supports both logical masking (zeroing the output of selected modules) and
structural pruning (physically replacing a module with an identity or
zero-output placeholder).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base_pruner import BasePruner
from ..utils import count_parameters


class ModulePruner(BasePruner):
    """Prune entire modules (transformer layers, MLP blocks, etc.).

    **Logical masking** registers a forward hook on each targeted module that
    zeros its output, effectively removing its contribution without modifying
    the model graph.

    **Structural pruning** replaces targeted modules with either an
    ``nn.Identity`` (pass-through) or a ``ZeroModule`` (emits zeros of the
    correct shape). This actually removes the module's parameters from the
    computation graph and can yield memory / speed benefits.

    Configuration keys
    ------------------
    - ``target_modules`` (list[str]): Module names to prune, e.g.
      ``["model.layers.26", "model.layers.27"]`` for Qwen2.
    - ``mode`` (str): ``"masking"`` or ``"structural"``.
    - ``structural_replacement`` (str): ``"identity"`` (default) or
      ``"zero"`` — what to use when structurally pruning.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_masking(self) -> None:
        self._validate_targets()
        print(f"Applying logical masking to {len(self.config['target_modules'])} module(s) …")
        self._handles = []
        for name in self.config["target_modules"]:
            module = self._find_module(name)
            if self.config.get("masking_replacement", "zero") == "identity":
                def bypass(module, inputs, output):
                    hidden = inputs[0]
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                handle = module.register_forward_hook(bypass)
            else:
                handle = module.register_forward_hook(self._make_zero_hook(name))
            self._handles.append(handle)
        self._masking_applied = True
        print("[OK] Logical masking applied.")

    def apply_structural(self) -> None:
        self._validate_targets()
        replacement_type = self.config.get("structural_replacement", "identity")
        print(
            f"Applying structural pruning ({replacement_type}) "
            f"to {len(self.config['target_modules'])} module(s) …"
        )
        self._replacements = {}
        before_params = count_parameters(self.model)
        for name in self.config["target_modules"]:
            parent, child_name = self._resolve_parent(name)
            original = getattr(parent, child_name)
            if replacement_type == "identity":
                replacement = _TransformerBlockIdentity()
            else:
                replacement = _TransformerBlockZero()
            setattr(parent, child_name, replacement)
            self._replacements[name] = original
        after_params = count_parameters(self.model)
        print(f"  Parameters before: {before_params:,}")
        print(f"  Parameters after:  {after_params:,}")
        print(f"  Removed:           {before_params - after_params:,}")
        self._structural_applied = True
        print("[OK] Structural pruning applied.")

    def revert(self) -> None:
        # Remove forward hooks
        if hasattr(self, "_handles"):
            for h in self._handles:
                h.remove()
            self._handles = []
        # Restore replaced modules
        if hasattr(self, "_replacements"):
            for name, original in self._replacements.items():
                parent, child_name = self._resolve_parent(name)
                setattr(parent, child_name, original)
            self._replacements = {}
        super().revert()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_targets(self) -> None:
        targets = self.config.get("target_modules")
        if not targets:
            raise ValueError("config['target_modules'] must be a non-empty list of module names.")
        for name in targets:
            self._find_module(name)

    def _find_module(self, name: str) -> nn.Module:
        """Look up a module by dotted name (e.g. 'model.layers.0.mlp')."""
        module = self.model
        for part in name.split("."):
            if part.isdigit():
                part = int(part)
            try:
                module = module[part] if isinstance(part, int) and isinstance(module, nn.ModuleList) else getattr(module, part)
            except (AttributeError, IndexError, TypeError):
                raise AttributeError(f"Module '{name}' not found in model.")
        return module

    def _resolve_parent(self, name: str):
        """Return (parent_module, child_attribute_name) for a dotted name."""
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            if part.isdigit():
                part = int(part)
            try:
                parent = parent[part] if isinstance(part, int) and isinstance(parent, nn.ModuleList) else getattr(parent, part)
            except (AttributeError, IndexError, TypeError):
                raise AttributeError(f"Module '{'.'.join(parts[:-1])}' not found while resolving parent of '{name}'.")
        return parent, parts[-1]

    @staticmethod
    def _make_zero_hook(name: str):
        """Return a hook that zeros the output of a module."""
        def hook(module, input, output):
            # Handle tuple outputs (e.g., attentions with cache)
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return hook


class _TransformerBlockIdentity(nn.Module):
    """Drop-in identity replacement for a transformer block.

    Blocks receive hidden states and may receive cache/mask keyword
    arguments. This replacement accepts those arguments and passes the
    hidden states through unchanged, returning a tuple compatible with
    common decoder block output formats.
    """

    def forward(self, hidden_states, *args, **kwargs):
        # Preserve the hidden states; the optional second item is a cache.
        return (hidden_states, None)


class _TransformerBlockZero(nn.Module):
    """Zero-output replacement for a transformer block.

    Emits zeros of the same shape as the input hidden states, plus a
    ``None`` for the optional cache output.
    """

    def forward(self, hidden_states, *args, **kwargs):
        return (torch.zeros_like(hidden_states), None)
