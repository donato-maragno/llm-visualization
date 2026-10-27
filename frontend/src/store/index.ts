import { writable, derived } from 'svelte/store';

// Upstream visualization stores, supplied exclusively by the local Python lab.
export const pipeline = writable<any>(null);
export const pipelinePair = writable<any>(null);
export const topology = writable<any>(null);
export const blockIdx = writable(0);
export const attentionHeadIdx = writable(0);
export const attentionHeadIdxTemp = writable(0);
export const queryIdx = writable(0);
export const diagramWidth = writable(1600);
export const tokens = writable<string[]>([]);
export const tokenIds = writable<number[]>([]);
export const modelMeta = writable({layer_num: 12, attention_head_num: 12, dimension: 768});
export const modelData = writable<any>({outputs: {}, probabilities: []});
export const predictedToken = writable<any>(undefined);
export const expandedBlock = writable<any>({id: null});
export const vectorHeight = writable(16);
export const headContentHeight = writable(480);
export const rootRem = 16;
export const headGap = {x: 5, y: 8, scale: 0};
export const predictedColor = '#9333ea';
export const minVectorHeight = 12;
export const maxVectorHeight = 30;
export const maxVectorScale = 3.4;
export const isOnAnimation = writable(false);
export const isOnBlockTransition = writable(false);
export const isExpandOrCollapseRunning = writable(false);
export const isBoundingBoxActive = writable(false);
export const isTextbookOpen = writable(false);
export const textbookCurrentPage = writable(0);
export const textbookCurrentPageId = writable('');
export const textbookPreviousPage = writable(-1);
export const userId = writable(null);
export const weightPopover = writable<any>(null);
export const tooltip = writable<any>(null);
export const hoveredMatrixCell = writable<any>({row: null, col: null});
export const highlightedIndex = writable<any>(null);
export const temperature = writable(1);
export const sampling = writable({type: 'top-k', value: Number.MAX_SAFE_INTEGER});

export const layer = derived([pipeline, blockIdx], ([$pipeline, $blockIdx]) => $pipeline?.layers[$blockIdx]);
export const sharedRanges = derived([pipelinePair, blockIdx], ([$pair, $blockIdx]) => {
  const ranges: Record<string, number> = {};
  if (!$pair) return ranges;
  for (const field of ['input', 'output', 'q', 'k', 'v', 'head_output', 'mlp_output', 'scores']) {
    ranges[field] = Math.max(0.000001, ...[$pair.a, $pair.b].flatMap(p =>
      p.layers[$blockIdx][field].flat(Infinity).map(Math.abs)));
  }
  for (const field of ['token', 'position', 'combined']) {
    ranges[field] = Math.max(0.000001, ...[$pair.a, $pair.b].flatMap(p =>
      p.embeddings[field].flat(Infinity).map(Math.abs)));
  }
  ranges['final-output'] = Math.max(0.000001, ...[$pair.a, $pair.b].flatMap(p =>
    p.layers.at(-1).output.flat(Infinity).map(Math.abs)));
  return ranges;
});
