"""Small, JSON-safe GPT-2 mask snapshots from the actual applied pruning state.

This describes topology and mask positions, not activations or measured speedups.
Never serialize full weight tensors to the browser.
"""


def gpt2_snapshot(model, variant, pruner):
    if model.config.model_type != "gpt2":
        return None
    count = model.config.n_layer
    heads = model.config.n_head
    saved = getattr(pruner, "_saved", {})
    technique = variant.technique if variant is not None else "none"
    strength = variant.strength if variant is not None else 0
    layers = []
    for index, block in enumerate(model.transformer.h):
        prefix = f"transformer.h.{index}"
        weights = []
        for name, param in block.mlp.named_parameters():
            key = f"{prefix}.mlp.{name}"
            positions = saved.get(key, (None, None))[0]
            weights.append({
                "name": key,
                "shape": list(param.shape),
                "total": param.numel(),
                "masked": int(positions.sum()) if positions is not None else 0,
                "sample": positions.reshape(-1)[:96].tolist() if positions is not None else [],
                "sample_values": param.detach().reshape(-1)[:96].float().cpu().tolist(),
            })
        # GPT-2 Conv1D stores c_proj as [input, output]. Each contiguous
        # group of input rows belongs to one head before mixing.
        projection = block.attn.c_proj.weight.detach()
        head_strengths = projection.float().abs().reshape(heads, -1).mean(dim=1).cpu().tolist()
        layers.append({
            "index": index, "name": prefix,
            "bypassed": technique == "layers" and index >= count - strength,
            "masked_heads": list(range(strength)) if technique == "heads" else [],
            "head_strengths": head_strengths,
            "mlp_weights": weights,
            "masked_weights": sum(w["masked"] for w in weights),
            "total_mlp_weights": sum(w["total"] for w in weights),
        })
    return {
        "model": "gpt2", "layer_count": count, "head_count": heads,
        "hidden_size": model.config.n_embd,
        "technique": technique, "strength": strength,
        "layers": layers,
    }
