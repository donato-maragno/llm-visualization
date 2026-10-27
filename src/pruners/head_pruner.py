"""
Attention-head pruning.

Supports GPT-2 (fused QKV), OPT (separate Q/K/V), and Qwen2 (GQA).

Qwen2 uses Grouped Query Attention: 12 Q heads but only 2 KV heads.
When pruning a Q head, we zero only that head's Q projection slice.
When pruning a KV head (index 0 or 1), we zero the corresponding slice
of K and V projections (each KV head covers 6 Q heads).

This module supports both masking and structural approaches.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base_pruner import BasePruner
from ..utils import count_parameters


class HeadPruner(BasePruner):
    """Prune individual attention heads in a transformer model.

    Configuration keys
    ------------------
    - head_indices : Spec for which heads to prune.
      Same formats as before (per-layer dict, global list, "even", etc.).

    - mode : "masking" or "structural".
    - zero_output_projection : Also zero output projection rows (structural).
    """

    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self._num_heads: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._num_layers: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._is_gqa: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_masking(self) -> None:
        self._discover_architecture()
        per_layer = self._normalize_head_spec()
        print("Applying logical masking to attention heads ...")
        self._handles = []

        for layer_idx, head_list in per_layer.items():
            block = self._get_transformer_block(layer_idx)
            attn = self._get_attention(block)
            out_proj = self._get_output_projection(attn)

            # Correct head masking: zero the head's concatenated output
            # BEFORE the output projection (o_proj / c_proj), so only the
            # selected heads' contributions are removed.
            handle = out_proj.register_forward_pre_hook(
                self._make_head_prehook(head_list, self._num_heads, self._head_dim)
            )
            self._handles.append(handle)

        self._masking_applied = True
        total_heads = sum(len(v) for v in per_layer.values())
        print(f"[OK] Masked {total_heads} head(s) across {len(per_layer)} layer(s).")
        if self._is_gqa:
            print(f"     Note: GQA model ({self._num_heads} Q heads, {self._num_kv_heads} KV heads).")
            print(f"     Pruning KV heads affects multiple Q heads.")

    def apply_structural(self) -> None:
        self._discover_architecture()
        per_layer = self._normalize_head_spec()
        zero_output = self.config.get("zero_output_projection", False)
        print(f"Applying structural head pruning (zero_output_proj={zero_output}) ...")

        total_pruned = 0
        for layer_idx, head_list in per_layer.items():
            block = self._get_transformer_block(layer_idx)
            attn = self._get_attention(block)
            self._zero_qkv_heads(attn, head_list)
            if zero_output:
                self._zero_output_projection(attn, head_list)
            total_pruned += len(head_list)

        self._structural_applied = True
        print(f"[OK] Structurally zeroed {total_pruned} head(s).")
        print(f"  Parameter count unchanged at {count_parameters(self.model):,}.")

    def revert(self) -> None:
        if hasattr(self, "_handles"):
            for h in self._handles:
                h.remove()
            self._handles = []
        super().revert()

    # ------------------------------------------------------------------
    # Architecture discovery
    # ------------------------------------------------------------------

    def _discover_architecture(self) -> None:
        cfg = self.model.config

        # num_layers
        for attr in ("num_hidden_layers", "n_layer"):
            if hasattr(cfg, attr):
                self._num_layers = getattr(cfg, attr)
                break
        if self._num_layers is None:
            raise RuntimeError("Cannot detect number of layers.")

        # num_heads
        for attr in ("num_attention_heads", "n_head"):
            if hasattr(cfg, attr):
                self._num_heads = getattr(cfg, attr)
                break
        if self._num_heads is None:
            raise RuntimeError("Cannot detect number of attention heads.")

        # GQA: num_key_value_heads (Qwen2, Llama3)
        if hasattr(cfg, "num_key_value_heads"):
            self._num_kv_heads = cfg.num_key_value_heads
            self._is_gqa = self._num_kv_heads < self._num_heads

        # hidden_size
        if hasattr(cfg, "hidden_size"):
            hidden = cfg.hidden_size
        elif hasattr(cfg, "n_embd"):
            hidden = cfg.n_embd
        else:
            raise RuntimeError("Cannot detect hidden size.")
        self._head_dim = hidden // self._num_heads

    def _get_transformer_block(self, layer_idx: int) -> nn.Module:
        """Return the nn.Module for a specific transformer layer.

        Supports GPT-2 (transformer.h), Qwen2 (model.layers),
        and OPT (model.decoder.layers).
        """
        # Qwen2, Llama
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        # GPT-2
        if hasattr(self.model, "transformer"):
            return self.model.transformer.h[layer_idx]
        # OPT
        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            return self.model.model.decoder.layers[layer_idx]
        raise RuntimeError(f"Unsupported architecture: {type(self.model).__name__}")

    def _get_attention(self, block: nn.Module) -> nn.Module:
        """Return the attention sub-module."""
        if hasattr(block, "self_attn"):       # Qwen2, Llama, OPT
            return block.self_attn
        if hasattr(block, "attn"):            # GPT-2
            return block.attn
        raise RuntimeError(f"Cannot find attention in block: {type(block).__name__}")

    # ------------------------------------------------------------------
    # Head-index normalisation
    # ------------------------------------------------------------------

    def _normalize_head_spec(self) -> Dict[int, List[int]]:
        """Convert head_indices config into {layer: [heads]} dict."""
        raw = self.config.get("head_indices")
        if raw is None:
            raise ValueError("head_indices is required.")

        if isinstance(raw, dict):
            out = {}
            for k, v in raw.items():
                idx = int(k)
                if not (0 <= idx < self._num_layers):
                    raise ValueError(f"Layer {idx} out of range [0, {self._num_layers}).")
                out[idx] = self._resolve_indices(v, self._num_heads, f"layer {idx} head")
            return out

        heads = self._resolve_indices(raw, self._num_heads, "head")
        return {i: heads for i in range(self._num_layers)}

    # ------------------------------------------------------------------
    # Masking (correct: zero the head's contribution before projection)
    # ------------------------------------------------------------------

    def _get_output_projection(self, attn: nn.Module) -> nn.Module:
        """Return the attention output projection (o_proj / c_proj / out_proj)."""
        for name in ("o_proj", "c_proj", "out_proj"):
            proj = getattr(attn, name, None)
            if proj is not None:
                return proj
        raise RuntimeError(f"Cannot find output projection in attention: {type(attn).__name__}")

    def _make_head_prehook(self, head_indices: List[int], num_heads: int, head_dim: int):
        """Return a pre-hook that zeros selected heads before the projection.

        The output projection receives the concatenated per-head outputs of
        shape (batch, seq, num_heads * head_dim).  Zeroing slices
        [h*head_dim : (h+1)*head_dim] removes exactly head h's contribution.
        """
        def prehook(module, input):
            x = input[0].clone()
            bsz, seq_len, _ = x.shape
            x = x.view(bsz, seq_len, num_heads, head_dim)
            x[:, :, head_indices, :] = 0.0
            x = x.view(bsz, seq_len, num_heads * head_dim)
            return (x,)

        return prehook

    # ------------------------------------------------------------------
    # Structural zeroing
    # ------------------------------------------------------------------

    def _zero_qkv_heads(self, attn: nn.Module, head_indices: List[int]) -> None:
        """Zero Q/K/V slices for pruned heads.

        GPT-2: fused c_attn.  Qwen2/OPT: separate q_proj, k_proj, v_proj.
        GQA: KV heads cover multiple Q heads — pruning a KV head
        (index 0 or 1) zeros a larger slice.
        """
        head_dim = self._head_dim
        num_heads = self._num_heads

        # -- Fused QKV (GPT-2) --
        if hasattr(attn, "c_attn"):
            hidden = num_heads * head_dim
            w = attn.c_attn.weight.data
            b = attn.c_attn.bias.data if attn.c_attn.bias is not None else None
            for h in head_indices:
                s, e = h * head_dim, (h + 1) * head_dim
                for offset in [0, hidden, 2 * hidden]:
                    w[s + offset : e + offset, :] = 0.0
                    if b is not None:
                        b[s + offset : e + offset] = 0.0
            return

        # -- Separate projections (Qwen2, OPT, LLaMA) --
        # Q head pruning
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is not None:
            for h in head_indices:
                s, e = h * head_dim, (h + 1) * head_dim
                q_proj.weight.data[s:e, :] = 0.0
                if q_proj.bias is not None:
                    q_proj.bias.data[s:e] = 0.0

        # KV head pruning (handles GQA)
        if self._is_gqa and self._num_kv_heads:
            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            heads_per_kv = num_heads // self._num_kv_heads
            for h in head_indices:
                kv_idx = h // heads_per_kv
                kv_dim = head_dim
                s, e = kv_idx * kv_dim, (kv_idx + 1) * kv_dim
                if k_proj is not None:
                    k_proj.weight.data[s:e, :] = 0.0
                    if k_proj.bias is not None:
                        k_proj.bias.data[s:e] = 0.0
                if v_proj is not None:
                    v_proj.weight.data[s:e, :] = 0.0
                    if v_proj.bias is not None:
                        v_proj.bias.data[s:e] = 0.0

    def _zero_output_projection(self, attn: nn.Module, head_indices: List[int]) -> None:
        head_dim = self._head_dim
        for proj_name in ("o_proj", "c_proj", "out_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue
            for h in head_indices:
                s, e = h * head_dim, (h + 1) * head_dim
                proj.weight.data[:, s:e] = 0.0
            return
