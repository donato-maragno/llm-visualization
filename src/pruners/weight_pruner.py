"""
Weight-level pruning.

Operates on individual parameter tensors rather than architectural
components.  Supports magnitude-based, random, and threshold pruning,
as well as custom per-tensor masks.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .base_pruner import BasePruner
from ..utils import count_nonzero_parameters, count_parameters


class WeightPruner(BasePruner):
    """Prune individual weights (elements of parameter tensors).

    **Logical masking** multiplies each targeted parameter by a binary mask
    on every forward pass via a ``pre_forward_hook``.  The mask is stored
    alongside the parameter so it participates in ``state_dict`` if desired.

    **Structural pruning** directly zeros the weights in-place (``param.data
    *= mask``).  The parameter shape is unchanged, so this is really "sparse
    structural" pruning — no memory savings unless followed by sparse-storage
    compression.  However, zeroed weights are guaranteed not to contribute.

    Configuration keys
    ------------------
    - ``strategy`` (str): One of ``"magnitude"``, ``"random"``, ``"threshold"``,
      or ``"custom"``.
    - ``sparsity`` (float, 0–1): Fraction of weights to prune (for magnitude
      and random strategies).
    - ``threshold`` (float): Absolute value below which weights are zeroed
      (for threshold strategy).
    - ``target_modules`` (list[str] | str): Module name pattern(s) whose
      parameters should be pruned.  Can be ``"all"`` for every parameter,
      or a list of dotted module names / glob-like patterns (e.g.
      ``["model.layers.*.mlp.*"]`` for Qwen2).
    - ``custom_mask_fn`` (Callable): If strategy is ``"custom"``, a function
      ``(name: str, param: torch.Tensor) -> torch.Tensor`` that returns a
      binary mask (same shape as param, dtype=bool or float).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_masking(self) -> None:
        self._validate_config()
        sparsity = self.config.get("sparsity", 0.5)

        print(f"Applying logical weight masking (strategy={self.config['strategy']}, sparsity={sparsity:.0%}) ...")

        # For inference, "masking" is implemented by zeroing the masked
        # weights in-place, but keeping a copy of the removed values so
        # we can restore them on revert().  This is memory-efficient:
        # only the PRUNED weights (not the full tensor) are stored.
        self._saved: Dict[str, tuple] = {}

        total = 0
        nonzero = 0
        for name, param in self._iter_target_params():
            mask = self._compute_mask(name, param).to(param.device)
            pruned_positions = (mask == 0)
            total += param.numel()
            nonzero += int((mask != 0).sum().item())

            if pruned_positions.any():
                # Save original values of the pruned positions for revert
                # Store recovery data on CPU to keep full-model pruning
                # practical on consumer GPUs with limited VRAM.
                self._saved[name] = (
                    pruned_positions.cpu(),
                    param.data[pruned_positions].detach().cpu().clone(),
                )
                # Zero them in-place
                param.data[pruned_positions] = 0.0

        self._masking_applied = True
        actual_sparsity = 1.0 - nonzero / total if total > 0 else 0.0
        print(f"[OK] Logical masking applied - {nonzero:,} / {total:,} weights retained ({actual_sparsity:.1%} pruned).")

    def apply_structural(self) -> None:
        self._validate_config()
        sparsity = self.config.get("sparsity", 0.5)

        print(f"Applying structural weight pruning (strategy={self.config['strategy']}, sparsity={sparsity:.0%}) …")

        before_total = count_parameters(self.model)
        for name, param in self._iter_target_params():
            mask = self._compute_mask(name, param)
            param.data *= mask.to(param.device, dtype=param.dtype)

        after_nonzero = count_nonzero_parameters(self.model)
        actual_sparsity = 1.0 - after_nonzero / before_total if before_total > 0 else 0.0
        print(f"[OK] Structural pruning applied — {after_nonzero:,} / {before_total:,} weights retained ({actual_sparsity:.1%} pruned).")
        self._structural_applied = True

    def revert(self) -> None:
        # Restore masked weights (masking mode)
        if hasattr(self, "_saved") and self._saved:
            params_by_name = dict(self.model.named_parameters())
            for name, (positions, original_values) in self._saved.items():
                if name in params_by_name:
                    param = params_by_name[name]
                    positions = positions.to(param.device)
                    param.data[positions] = original_values.to(param.device, dtype=param.dtype)
            self._saved = {}
        super().revert()

    def get_sparsity_report(self) -> Dict[str, float]:
        """Return per-layer sparsity after pruning.

        Returns:
            Dict mapping parameter names to their sparsity (0–1).
        """
        report = {}
        for name, param in self.model.named_parameters():
            total = param.numel()
            nonzero = torch.count_nonzero(param.data).item()
            report[name] = 1.0 - (nonzero / total) if total > 0 else 0.0
        return report

    # ------------------------------------------------------------------
    # Mask computation
    # ------------------------------------------------------------------

    def _compute_mask(self, name: str, param: nn.Parameter) -> torch.Tensor:
        """Generate a binary mask for a single parameter tensor."""
        strategy = self.config["strategy"]
        shape = param.data.shape
        device = param.device

        if strategy == "magnitude":
            return self._magnitude_mask(param, self.config["sparsity"])
        elif strategy == "random":
            return self._random_mask(shape, self.config["sparsity"], device)
        elif strategy == "threshold":
            threshold = self.config.get("threshold", 0.01)
            return self._threshold_mask(param, threshold)
        elif strategy == "custom":
            fn = self.config.get("custom_mask_fn")
            if fn is None:
                raise ValueError("custom_mask_fn must be provided for 'custom' strategy.")
            mask = fn(name, param)
            if mask.shape != param.shape:
                raise ValueError(
                    f"Custom mask for '{name}' has shape {mask.shape}, expected {param.shape}."
                )
            return mask
        else:
            raise ValueError(f"Unknown weight pruning strategy: {strategy!r}")

    @staticmethod
    def _magnitude_mask(param: nn.Parameter, sparsity: float) -> torch.Tensor:
        """Keep the top (1 - sparsity) weights by absolute magnitude.

        Selects a magnitude threshold in FP32 so FP16 CUDA weights work
        without quantile's dtype and tensor-size restrictions.
        """
        if sparsity <= 0:
            return torch.ones_like(param.data)
        if sparsity >= 1:
            return torch.zeros_like(param.data)
        abs_vals = param.data.abs()
        flat = abs_vals.flatten().float()
        k = min(flat.numel(), max(1, int(flat.numel() * sparsity) + 1))
        threshold = torch.kthvalue(flat, k).values
        return (abs_vals >= threshold).float()

    @staticmethod
    def _random_mask(shape: torch.Size, sparsity: float, device: torch.device) -> torch.Tensor:
        """Randomly retain (1 - sparsity) fraction of weights."""
        mask = torch.rand(shape, device=device) > sparsity
        return mask.float()

    @staticmethod
    def _threshold_mask(param: nn.Parameter, threshold: float) -> torch.Tensor:
        """Zero weights with absolute value below threshold."""
        return (param.data.abs() >= threshold).float()

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_param_mask_hook(mask: torch.Tensor):
        """Return a backward-hook-compatible grad multiplier.

        During forward, the hook multiplies the gradient by the mask so
        pruned weights stay zero even after optimizer steps.
        """
        def hook(grad):
            return grad * mask.to(grad.device, dtype=grad.dtype)
        return hook

    # ------------------------------------------------------------------
    # Target iteration
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        strategy = self.config.get("strategy")
        if strategy not in ("magnitude", "random", "threshold", "custom"):
            raise ValueError(f"Unknown weight pruning strategy: {strategy!r}")

    def _iter_target_params(self):
        """Yild (name, param) for parameters matching the target specification.

        Embedding tables (``embed_tokens``, ``wte``, ``lm_head``) are always
        skipped — pruning them is unusual and they are the largest tensors,
        which would otherwise cause out-of-memory on big models.
        """
        skip_keywords = ("embed_tokens", "wte", "lm_head", "wpe")

        def _should_skip(name: str) -> bool:
            return any(k in name for k in skip_keywords)

        targets = self.config.get("target_modules", "all")
        if targets == "all":
            for name, param in self.model.named_parameters():
                if not _should_skip(name):
                    yield name, param
            return

        if isinstance(targets, str):
            targets = [targets]

        for name, param in self.model.named_parameters():
            if _should_skip(name):
                continue
            for pattern in targets:
                if self._match_pattern(name, pattern):
                    yield name, param
                    break

    def _iter_params(self):
        """Yild all parameters matching the target specification (params only, no names)."""
        for _name, param in self._iter_target_params():
            yield param

    @staticmethod
    def _match_pattern(name: str, pattern: str) -> bool:
        """Simple glob-like matching with '*' wildcard support."""
        import re
        regex = re.escape(pattern).replace(r"\*", ".*")
        return re.fullmatch(regex, name) is not None
