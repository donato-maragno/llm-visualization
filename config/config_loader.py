"""
Configuration loader for pruning experiments.

Loads YAML configuration files and provides validation and defaults.
Configuration is kept separate from implementation so you can switch
between experiments by pointing at a different YAML file.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_config_dir() -> Path:
    """Return the absolute path to the ``config/`` directory."""
    return Path(__file__).resolve().parent


def get_pruning_configs_dir() -> Path:
    """Return the absolute path to ``config/pruning_configs/``."""
    return get_config_dir() / "pruning_configs"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return it as a Python dictionary.

    Args:
        path: Path to the YAML file (absolute or relative to cwd).

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(path):
        # Try relative to the config directory
        alt = get_config_dir() / path
        if alt.exists():
            path = str(alt)
        else:
            raise FileNotFoundError(f"Config file not found: {path}  (also tried {alt})")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at the top level, got {type(data).__name__}.")

    return data


def load_pruning_config(name: str) -> Dict[str, Any]:
    """Load a named pruning configuration from ``config/pruning_configs/``.

    Args:
        name: Config filename without the ``.yaml`` extension
            (e.g. ``"module_pruning"`` → loads ``module_pruning.yaml``).

    Returns:
        Parsed config dictionary.
    """
    path = get_pruning_configs_dir() / f"{name}.yaml"
    return load_yaml(str(path))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_pruning_config(config: Dict[str, Any]) -> None:
    """Basic validation of a pruning configuration dictionary.

    Raises ``ValueError`` if required keys are missing or values
    are obviously wrong.
    """
    required = ["type", "mode"]
    for key in required:
        if key not in config:
            raise ValueError(f"Pruning config must contain '{key}' key.")

    valid_types = ("module", "head", "weight")
    if config["type"] not in valid_types:
        raise ValueError(f"Pruning type must be one of {valid_types}, got {config['type']!r}.")

    valid_modes = ("masking", "structural")
    if config["mode"] not in valid_modes:
        raise ValueError(f"Pruning mode must be one of {valid_modes}, got {config['mode']!r}.")


# ---------------------------------------------------------------------------
# Factory: create a pruner from a config
# ---------------------------------------------------------------------------

def create_pruner_from_config(model, config: Dict[str, Any]):
    """Instantiate the appropriate pruner class from a config dict.

    Args:
        model: The loaded model.
        config: Pruning configuration dict (must have a ``"type"`` key).

    Returns:
        An instance of ``ModulePruner``, ``HeadPruner``, or ``WeightPruner``.

    Raises:
        ValueError: If the config is invalid.
    """
    validate_pruning_config(config)

    pruner_type = config["type"]

    if pruner_type == "module":
        from src.pruners.module_pruner import ModulePruner
        return ModulePruner(model, config)

    if pruner_type == "head":
        from src.pruners.head_pruner import HeadPruner
        return HeadPruner(model, config)

    if pruner_type == "weight":
        from src.pruners.weight_pruner import WeightPruner
        return WeightPruner(model, config)

    raise ValueError(f"Unknown pruner type: {pruner_type!r}")
