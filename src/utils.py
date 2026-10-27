"""
Utility functions for the pruning experiments framework.

Provides helpers for parameter counting, timing, memory tracking,
and other common operations used across pruning modules.
"""

import time
from contextlib import contextmanager
from typing import Optional

import torch
import psutil
import os


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only parameters that require gradients.

    Returns:
        Total number of (trainable) parameters.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_nonzero_parameters(model: torch.nn.Module) -> int:
    """Count parameters with non-zero values (useful after weight pruning).

    Args:
        model: PyTorch model.

    Returns:
        Number of non-zero parameter elements.
    """
    total = 0
    for p in model.parameters():
        total += torch.count_nonzero(p).item()
    return total


def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model memory footprint in megabytes.

    Uses float32 (4 bytes per element) as the default precision.

    Args:
        model: PyTorch model.

    Returns:
        Approximate size in megabytes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(name: str = "Operation", verbose: bool = True):
    """Context manager to measure and report elapsed wall-clock time.

    Usage:
        with timer("inference"):
            output = model(input_ids)

    Args:
        name: Label printed alongside the elapsed time.
        verbose: If False, suppress printing (elapsed is still returned).
    """
    start = time.perf_counter()
    elapsed = 0.0
    try:
        yield lambda: elapsed
    finally:
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"[{name}] elapsed: {elapsed:.4f} s")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def get_process_memory_mb() -> float:
    """Return the current process's resident memory in megabytes."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


@contextmanager
def memory_tracker(name: str = "Operation", verbose: bool = True):
    """Context manager that reports the change in process memory.

    Usage:
        with memory_tracker("load model"):
            model = AutoModel.from_pretrained(...)
    """
    before = get_process_memory_mb()
    try:
        yield
    finally:
        after = get_process_memory_mb()
        delta = after - before
        if verbose:
            print(f"[{name}] memory delta: {delta:+.2f} MB  (before={before:.2f}, after={after:.2f})")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model structure inspection
# ---------------------------------------------------------------------------

def print_model_structure(
    model: torch.nn.Module,
    max_depth: int = 3,
    indent: int = 0,
    prefix: str = "",
) -> None:
    """Recursively print a model's module hierarchy for inspection.

    Args:
        model: The PyTorch model to inspect.
        max_depth: Maximum nesting depth to print.
        indent: Internal – current indentation level.
        prefix: Internal – accumulated module name prefix.
    """
    if indent > max_depth:
        return

    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        n_params = sum(p.numel() for p in child.parameters())
        print(f"{'  ' * indent}{full_name}  ({child.__class__.__name__}, params={n_params:,})")
        if indent < max_depth:
            print_model_structure(child, max_depth, indent + 1, full_name)