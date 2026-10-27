"""Bounded, measured GPT-2 forward-pass snapshots for post-run comparisons.

Both models receive exactly the same prompt. Only small vector samples and
attention matrices are serialized; no hooks remain installed after capture.
"""

import math

import torch


MAX_PROMPT_TOKENS = 32
VECTOR_SAMPLE = 8


def _values(tensor):
    return tensor.detach().float().cpu().round(decimals=6).tolist()


def prompt_inputs(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(next(model.parameters()).device)
    count = inputs.input_ids.shape[1]
    limit = min(MAX_PROMPT_TOKENS, model.config.max_position_embeddings - 1)
    if not 0 < count <= limit:
        raise ValueError(f"Use a shorter prompt: the pipeline supports 1–{limit} tokens; this prompt has {count}. No input was truncated.")
    return inputs


@torch.inference_mode()
def capture_pipeline(model, tokenizer, prompt, topology=None):
    if model.config.model_type != "gpt2":
        raise ValueError("Text-pipeline inspection currently supports GPT-2 only.")
    inputs = prompt_inputs(model, tokenizer, prompt)
    ids = inputs.input_ids[0]
    count = len(ids)
    heads = model.config.n_head
    head_dim = model.config.n_embd // heads
    layers = [{} for _ in model.transformer.h]
    handles = []
    previous_attention = model.config._attn_implementation

    def qkv_hook(index):
        def capture(module, args, output):
            q, k, v = [part[0].reshape(count, heads, head_dim).transpose(0, 1)
                       for part in output.split(model.config.n_embd, dim=-1)]
            attention = model.transformer.h[index].attn
            # Match GPT-2 eager attention's precision, including optional upcast.
            score_q = q.float() if attention.reorder_and_upcast_attn else q
            score_k = k.float() if attention.reorder_and_upcast_attn else k
            scores = score_q @ score_k.transpose(-1, -2)
            if attention.scale_attn_weights:
                scores /= math.sqrt(head_dim)
            if attention.scale_attn_by_inverse_layer_idx:
                scores /= index + 1
            layers[index].update(
                q=_values(q[..., :VECTOR_SAMPLE]), k=_values(k[..., :VECTOR_SAMPLE]),
                v=_values(v[..., :VECTOR_SAMPLE]), scores=_values(scores),
            )
        return capture

    def head_hook(index):
        def capture(module, args):
            # Registered after the pruner: these are the actual contributions,
            # including zeroed head slices, entering the output projection.
            value = args[0][0].reshape(count, heads, head_dim).transpose(0, 1)
            layers[index]["head_output"] = _values(value[..., :VECTOR_SAMPLE])
        return capture

    def mlp_hook(index):
        def capture(module, args, output):
            layers[index]["mlp_output"] = _values(output[0, :, :VECTOR_SAMPLE])
        return capture

    def block_hook(index):
        def capture(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            layers[index]["input"] = _values(args[0][0, :, :VECTOR_SAMPLE])
            layers[index]["output"] = _values(value[0, :, :VECTOR_SAMPLE])
        return capture

    try:
        model.set_attn_implementation("eager")
        for index, block in enumerate(model.transformer.h):
            handles.extend([
                block.attn.c_attn.register_forward_hook(qkv_hook(index)),
                block.attn.c_proj.register_forward_pre_hook(head_hook(index)),
                block.mlp.register_forward_hook(mlp_hook(index)),
                block.register_forward_hook(block_hook(index)),
            ])
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        token_vectors = model.transformer.wte(ids)
        position_vectors = model.transformer.wpe(torch.arange(count, device=ids.device))
        probabilities = outputs.logits[0].float().softmax(dim=-1)
        top_values, top_ids = probabilities.topk(min(8, probabilities.shape[-1]), dim=-1)
        predictions = []
        for position in range(count):
            candidates = [{"id": int(token_id), "text": tokenizer.decode([int(token_id)]),
                           "probability": float(probability)}
                          for token_id, probability in zip(top_ids[position], top_values[position])]
            predictions.append({"candidates": candidates, "other_probability": max(0., 1. - sum(c["probability"] for c in candidates))})
        for index, layer in enumerate(layers):
            state = topology["layers"][index] if topology else {}
            layer.update(index=index, attention=_values(outputs.attentions[index][0]),
                         bypassed=state.get("bypassed", False), masked_heads=state.get("masked_heads", []),
                         masked_weights=state.get("masked_weights", 0), total_mlp_weights=state.get("total_mlp_weights", 0))
        return {
            "schema": "gpt2-forward-v1", "prompt": prompt,
            "tokens": [{"id": int(token_id), "text": tokenizer.decode([int(token_id)]), "position": index}
                       for index, token_id in enumerate(ids)],
            "hidden_size": model.config.n_embd, "head_count": heads, "head_dim": head_dim,
            "vector_sample_size": min(VECTOR_SAMPLE, head_dim),
            "embeddings": {"token": _values(token_vectors[:, :VECTOR_SAMPLE]),
                           "position": _values(position_vectors[:, :VECTOR_SAMPLE]),
                           "combined": _values((token_vectors + position_vectors)[:, :VECTOR_SAMPLE])},
            "layers": layers, "predictions": predictions,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.set_attn_implementation(previous_attention)
