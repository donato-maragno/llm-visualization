import { get } from 'svelte/store';
import * as store from './store';

export function notify(type, values = {}) {
  if (window.parent !== window) window.parent.postMessage({type: `llm-visualizer:${type}`, ...values}, window.location.origin);
}

export function navigate(axis, delta) { notify('navigate', {axis, delta}); }
export function inspect(drawer) { notify('inspect', {drawer}); }

export function receive({a, b, view, layer, head, query, topology, width}) {
  const p = view === 'original' ? a : b;
  store.pipelinePair.set({a, b});
  store.pipeline.set(p);
  store.topology.set(topology);
  store.modelMeta.set({layer_num: p.layers.length, attention_head_num: p.head_count, dimension: p.hidden_size});
  store.tokens.set(p.tokens.map(t => t.text));
  store.tokenIds.set(p.tokens.map(t => t.id));
  store.blockIdx.set(layer);
  store.attentionHeadIdx.set(head);
  store.attentionHeadIdxTemp.set(head);
  store.queryIdx.set(query);
  store.diagramWidth.set(Math.max(1600, width || 1600));
  const outputs = {};
  p.layers.forEach((block, index) => {
    block.scores.forEach((scores, h) => {
      const prefix = `block_${index}_attn_head_${h}`;
      outputs[`${prefix}_attn`] = {data: scores};
      outputs[`${prefix}_attn_masked`] = {data: scores.map((row, i) => row.map((v, j) => j > i ? -Infinity : v))};
      outputs[`${prefix}_attn_dropout`] = {data: block.attention[h]};
    });
  });
  const probabilities = p.predictions[query].candidates.map((candidate, rank) => ({
    tokenId: candidate.id, token: candidate.text, rank,
    probability: candidate.probability,
    // log p is a normalized logit; displayed as log p, never as raw model logits.
    logit: Math.log(candidate.probability),
    scaledLogit: Math.log(candidate.probability),
    topKLogit: Math.log(candidate.probability)
  }));
  store.modelData.set({outputs, probabilities, otherProbability: p.predictions[query].other_probability});
  store.predictedToken.set(probabilities[0]);
}

// Keep upstream link/operation interactions useful without loading a second app
// or its textbook/analytics state.
store.weightPopover.subscribe(value => {
  if (value) inspect({qkv: 'qkv', attention: 'attention', mlpUp: 'mlp', mlpDown: 'mlp', softmax: 'probabilities'}[value] || 'mlp');
});

export function selectedState() {
  return {layer: get(store.blockIdx), head: get(store.attentionHeadIdx)};
}
