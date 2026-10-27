"""
Abstract base class for all pruning strategies.

Defines the common interface that every pruner must implement:
    - apply_masking()   → logical (runtime) masking
    - apply_structural()→ physical removal of components
    - revert()          → restore original model state
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BasePruner(ABC):
    """Abstract base for pruning strategies.

    Subclasses must implement `apply_masking` and `apply_structural`.

    Attributes:
        model: The model being pruned.
        config: Free-form configuration dictionary for this pruner.
        _original_state: State dict snapshot taken before any modifications,
                         used by `revert()` to restore the original model.
    """

    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.config = config or {}
        self._original_state: Optional[Dict[str, Any]] = None
        self._masking_applied = False
        self._structural_applied = False

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """Save a deep copy of the model's state_dict so we can revert later.

        Call this once before applying any pruning.
        """
        # Keep backups in host RAM so large CUDA models do not need a second
        # full copy of their weights in VRAM while pruning.
        self._original_state = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }

    def revert(self) -> None:
        """Restore the model to its pre-pruning state.

        If a snapshot was taken (needed for structural pruning), load it.
        Otherwise just reset flags — masking-mode pruners remove their own
        hooks / restore their own weights in their `revert()` overrides.
        """
        if self._original_state is not None:
            self.model.load_state_dict(self._original_state)
            self._original_state = None
        self._masking_applied = False
        self._structural_applied = False

    # ------------------------------------------------------------------
    # Pruning interface (to be overridden)
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_masking(self) -> None:
        """Apply logical masking so pruned components contribute zero.

        After masking the model architecture is unchanged (same output
        shapes, same parameter count), but selected components are
        prevented from influencing the computation.
        """
        ...

    @abstractmethod
    def apply_structural(self) -> None:
        """Physically remove or replace pruned components in the model.

        Structural pruning changes the model architecture permanently —
        the parameter count drops and inference may become faster.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers shared across pruners
    # ------------------------------------------------------------------

    def _resolve_indices(
        self, spec: Any, max_index: int, label: str = "index"
    ) -> list[int]:
        """Convert a user-facing specification into a concrete list of indices.

        Accepted formats:
            - int                → single index
            - list[int]          → explicit list
            - "all"              → every index in [0, max_index)
            - "even" / "odd"     → even / odd indices
            - slice string       → e.g. "0:6", "::2"

        Args:
            spec: The user specification.
            max_index: Upper bound (exclusive) for valid indices.
            label: Human-readable label for error messages.

        Returns:
            Sorted list of unique integer indices.

        Raises:
            ValueError: If the spec cannot be parsed or indices are
                out of range.
        """
        if isinstance(spec, int):
            indices = [spec]
        elif isinstance(spec, list):
            indices = spec
        elif spec == "all":
            indices = list(range(max_index))
        elif spec == "even":
            indices = list(range(0, max_index, 2))
        elif spec == "odd":
            indices = list(range(1, max_index, 2))
        elif isinstance(spec, str) and ":" in spec:
            parts = [int(p) if p.strip() else None for p in spec.split(":")]
            indices = list(range(*slice(*parts).indices(max_index)))
        else:
            raise ValueError(
                f"Invalid {label} specification: {spec!r}. "
                f"Use int, list[int], 'all', 'even', 'odd', or a slice string like '0:6'."
            )

        # Validate
        for idx in indices:
            if not (0 <= idx < max_index):
                raise ValueError(
                    f"{label} {idx} out of range [0, {max_index})."
                )
        return sorted(set(indices))
