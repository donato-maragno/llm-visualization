# Transformer Explainer visualization

Visualization components in `src/components/`, the D3 gradient/opacity constants,
matrix utilities, GSAP integration, and layout conventions are adapted from:

- https://github.com/poloclub/transformer-explainer
- Commit: `bfe50afba10b9b560b84143ee1107d977defa74f`
- License: MIT, reproduced in `LICENSE.upstream` and
  `../web/third-party/transformer-explainer-LICENSE.txt`.

This is a reuse of the upstream Svelte/D3 visualization, not an iframe pointing
to their website. Assets and compiled components are served locally.

## Local changes

- Replace upstream ONNX inference, cached examples and analytics with a
  same-origin bridge to the Python lab's measured GPT-2 snapshots.
- Keep stacked attention cards, D3 flow ribbons, residual operations, expandable
  embedding/attention/probability views, and original vector-column layout.
- Add original/pruned selection, layer/head wheel navigation and layer arrows.
- Show real vector samples instead of the upstream random decorative textures.
  Unrecorded MLP intermediates remain clearly labeled architectural placeholders.
- Overlay actual block bypasses, zeroed head contributions and MLP weight masks.
- Keep full-vocabulary probabilities unrenormalized. The probability expansion
  uses explicitly labeled `log p` rather than inventing unavailable raw logits;
  it does not expose upstream temperature or top-k sampling controls.
- Route conceptual/weight inspection to the lab's existing measured inspector;
  do not include upstream random example weight matrices or analytics.
- Compile the frontend into `../web/transformer/`; Python users do not need npm.

## Rebuild

With Node.js 20.19+ or 22.12+ installed:

```sh
cd frontend
npm ci
npm run build
```

Commit both source changes and the generated `web/transformer/` assets.
