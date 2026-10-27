"""Bounded transformer measurements tied to the exact tested problem prompt.

Capture a normal forward pass without requesting quadratic attention tensors.
Small attention submatrices are reconstructed from measured Q/K using ALL
causal keys for normalization, never by renormalizing a cropped token window.
"""

import math

import torch
from .pruning_targets import bypassed_layer_indices

from .judge_eval import prepare_generation_inputs
from .text_pipeline import _values, VECTOR_SAMPLE


MAX_DISPLAY_TOKENS = 16


@torch.inference_mode()
def capture_question_pipeline(model, tokenizer, problem, variant=None, pruner=None):
    model_type = model.config.model_type
    if model_type not in ("gpt2", "qwen2"):
        raise ValueError("Question pipelines support GPT-2 and Qwen2 models.")
    prompt, inputs = prepare_generation_inputs(model, tokenizer, problem.prompt(), max_new_tokens=None)
    ids = inputs.input_ids[0]
    count = len(ids)
    original_count = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.shape[1]
    if not count:
        raise ValueError("The question produced an empty model input.")
    # Evenly sample the full input, including its first and last token. The model
    # still receives its complete prepared input, not these display samples.
    positions = torch.linspace(0, count - 1, min(count, MAX_DISPLAY_TOKENS), device=ids.device).long().unique()
    displayed = len(positions)
    is_gpt2 = model_type == "gpt2"
    blocks = model.transformer.h if is_gpt2 else model.model.layers
    heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // heads
    layers = [{} for _ in blocks]
    handles, pending = [], {}
    technique = variant.technique if variant else "none"
    strength = variant.strength if variant else 0

    def record_attention(index, q, k, v, scaling, window=None):
        # q/k/v: [heads, full_sequence, head_dim], after RoPE for Qwen.
        query = q.index_select(1, positions)
        scores = (query.float() @ k.float().transpose(-1, -2)) * scaling
        allowed = torch.arange(count, device=ids.device)[None, :] <= positions[:, None]
        if window is not None:
            allowed &= torch.arange(count, device=ids.device)[None, :] > positions[:, None] - window
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        visible_attention = weights.index_select(2, positions)
        layers[index].update(
            q=_values(query[..., :VECTOR_SAMPLE]),
            k=_values(k.index_select(1, positions)[..., :VECTOR_SAMPLE]),
            v=_values(v.index_select(1, positions)[..., :VECTOR_SAMPLE]),
            scores=_values(scores.index_select(2, positions)), attention=_values(visible_attention),
            omitted_attention=_values((1 - visible_attention.sum(-1)).clamp(0, 1)),
        )

    def gpt2_qkv(index):
        def capture(module, args, output):
            q, k, v = [part[0].reshape(count, heads, head_dim).transpose(0, 1)
                       for part in output.split(hidden_size, dim=-1)]
            attention = blocks[index].attn
            scaling = 1 / math.sqrt(head_dim) if attention.scale_attn_weights else 1
            if attention.scale_attn_by_inverse_layer_idx:
                scaling /= index + 1
            record_attention(index, q, k, v, scaling)
        return capture

    def qwen_projection(index, kind):
        def capture(module, args, output):
            pending.setdefault(index, {})[kind] = output
        return capture

    def qwen_attention(index):
        def capture(module, args, kwargs, output):
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
            projections = pending.pop(index)
            kv_heads = model.config.num_key_value_heads
            q = projections["q"].reshape(1, count, heads, head_dim).transpose(1, 2)
            k = projections["k"].reshape(1, count, kv_heads, head_dim).transpose(1, 2)
            v = projections["v"].reshape(1, count, kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *kwargs["position_embeddings"])
            k, v = repeat_kv(k, heads // kv_heads), repeat_kv(v, heads // kv_heads)
            record_attention(index, q[0], k[0], v[0], module.scaling, module.sliding_window)
        return capture

    def head_input(index):
        def capture(module, args):
            # Runs after pruning prehooks, so masked head samples are truly zero.
            values = args[0][0].index_select(0, positions).reshape(displayed, heads, head_dim).transpose(0, 1)
            layers[index]["head_output"] = _values(values[..., :VECTOR_SAMPLE])
        return capture

    def mlp_output(index):
        def capture(module, args, output):
            layers[index]["mlp_output"] = _values(output[0].index_select(0, positions)[:, :VECTOR_SAMPLE])
        return capture

    def block_output(index):
        def capture(module, args, output):
            values = output[0] if isinstance(output, tuple) else output
            layers[index]["input"] = _values(args[0][0].index_select(0, positions)[:, :VECTOR_SAMPLE])
            layers[index]["output"] = _values(values[0].index_select(0, positions)[:, :VECTOR_SAMPLE])
        return capture

    try:
        for index, block in enumerate(blocks):
            attention = block.attn if is_gpt2 else block.self_attn
            if is_gpt2:
                handles.append(attention.c_attn.register_forward_hook(gpt2_qkv(index)))
            else:
                for kind in ("q", "k", "v"):
                    handles.append(getattr(attention, f"{kind}_proj").register_forward_hook(qwen_projection(index, kind)))
                handles.append(attention.register_forward_hook(qwen_attention(index), with_kwargs=True))
            projection = attention.c_proj if is_gpt2 else attention.o_proj
            handles.extend([
                projection.register_forward_pre_hook(head_input(index)),
                block.mlp.register_forward_hook(mlp_output(index)),
                block.register_forward_hook(block_output(index)),
            ])
        outputs = model(**inputs, use_cache=False, return_dict=True)
        embedding = model.transformer.wte(ids) if is_gpt2 else model.model.embed_tokens(ids)
        embeddings = {"token": _values(embedding.index_select(0, positions)[:, :VECTOR_SAMPLE])}
        if is_gpt2:
            position_embedding = model.transformer.wpe(torch.arange(count, device=ids.device))
            embeddings.update(position=_values(position_embedding.index_select(0, positions)[:, :VECTOR_SAMPLE]),
                              combined=_values((embedding + position_embedding).index_select(0, positions)[:, :VECTOR_SAMPLE]))
        predictions = []
        probabilities = outputs.logits[0].index_select(0, positions).float().softmax(-1)
        top_values, top_ids = probabilities.topk(min(8, probabilities.shape[-1]), dim=-1)
        for values, tokens in zip(top_values, top_ids):
            candidates = [{"id": int(token), "text": tokenizer.decode([int(token)]), "probability": float(value)}
                          for value, token in zip(values, tokens)]
            predictions.append({"candidates": candidates, "other_probability": max(0., 1. - sum(c["probability"] for c in candidates))})
        for index, (block, layer) in enumerate(zip(blocks, layers)):
            layer.update(index=index, bypassed=index in bypassed_layer_indices(variant, len(blocks)),
                         masked_heads=list(range(min(strength, heads - 1))) if technique == "heads" else [],
                         masked_weights=0, total_mlp_weights=sum(p.numel() for p in block.mlp.parameters()))
            if technique in ("magnitude", "random"):
                prefix = f"transformer.h.{index}.mlp." if is_gpt2 else f"model.layers.{index}.mlp."
                layer["masked_weights"] = sum(int(positions.sum()) for name, (positions, _) in getattr(pruner, "_saved", {}).items()
                                              if name.startswith(prefix))
        return {
            "schema": "question-forward-v1", "problem_id": problem.id, "problem_title": problem.title,
            "model_type": model_type, "prompt": prompt, "input_token_count": count,
            "input_text": tokenizer.decode(ids.tolist(), skip_special_tokens=False),
            "prompt_truncated": original_count > count, "original_input_token_count": original_count,
            "sampled_positions": len(positions) < count, "position_encoding": "learned" if is_gpt2 else "rotary",
            "attention_source": "recomputed_from_measured_qk_full_context",
            "tokens": [{"id": int(ids[position]), "text": tokenizer.decode([int(ids[position])]), "position": int(position)}
                       for position in positions],
            "hidden_size": hidden_size, "head_count": heads, "head_dim": head_dim,
            "vector_sample_size": min(VECTOR_SAMPLE, head_dim), "embeddings": embeddings,
            "layers": layers, "predictions": predictions,
        }
    finally:
        for handle in handles:
            handle.remove()
        pending.clear()
