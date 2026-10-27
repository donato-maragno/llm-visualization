<script>
  import { onMount, tick } from 'svelte';
  import { pipeline, layer, blockIdx, attentionHeadIdx, topology } from './store';
  import { inspect } from './bridge';
  let paths = [];
  let contour = null;
  let svg;
  $: currentTopology = $topology?.layers?.[$blockIdx];
  $: weights = currentTopology?.mlp_weights?.find(weight => weight.name.endsWith('c_fc.weight'));
  $: maskedHead = $layer?.masked_heads.includes($attentionHeadIdx);
  async function redraw() {
    await tick();
    if (!$layer?.bypassed || !svg) { paths = []; contour = null; return; }
    const starts = [...document.querySelectorAll('.block-steps.main .qkv .embedding-column .vector')];
    const ends = [...document.querySelectorAll('.block-steps.main .mlp .mlp-out-column .vector')];
    const rect = svg.getBoundingClientRect();
    const block = document.querySelector('.blocks').getBoundingClientRect();
    const headBottom = document.querySelector('.head-content').getBoundingClientRect().bottom - rect.top + 8 * ($pipeline.head_count - 1);
    contour = {x: block.left - rect.left - 10, y: block.top - rect.top + 42,
      width: block.width + 12, height: headBottom - (block.top - rect.top + 42) + 15};
    const floor = headBottom + 28;
    paths = starts.map((node, i) => {
      const a = node.getBoundingClientRect(), b = ends[i]?.getBoundingClientRect();
      if (!b) return '';
      const x1 = a.right - rect.left, y1 = a.top - rect.top + a.height / 2;
      const x2 = b.left - rect.left, y2 = b.top - rect.top + b.height / 2;
      return `M${x1},${y1} H${x1 + 8 + i * 2} V${floor + i * 2} H${x2 - 8 - i * 2} V${y2} H${x2}`;
    });
  }
  $: if ($pipeline && $blockIdx >= 0) redraw();
  onMount(() => {
    const observer = new ResizeObserver(redraw);
    observer.observe(document.querySelector('.nodes'));
    document.querySelectorAll('.blocks,.head-content,.blocks .resize-watch').forEach(node => observer.observe(node));
    return () => observer.disconnect();
  });
</script>

<svg class="pruning-overlay" bind:this={svg} aria-label="Applied pruning overlays">
  {#if contour}
    <g class="bypassed-layer-contour" data-bypassed-layer={$blockIdx + 1} aria-label={`Layer ${$blockIdx + 1} bypassed; transparent orange dashed contour`}>
      <rect x={contour.x} y={contour.y} width={contour.width} height={contour.height} rx="10" />
      <path class="bypassed-layer-depth" d={`M${contour.x + contour.width},${contour.y + 12} l12,10 v${contour.height - 20} l-12,-2 M${contour.x + 12},${contour.y + contour.height} l10,12 h${contour.width - 10}`} />
      <text x={contour.x + 14} y={contour.y + 18}>Layer {$blockIdx + 1} · bypassed</text>
    </g>
  {/if}
  {#each paths as path}<path class="bypass-route" d={path} />{/each}
</svg>
<div class="pruning-summary" aria-live="polite">
  {#if $layer?.bypassed}<span class="bypass-label">Layer {$blockIdx + 1} bypassed · effective output = input (internal computation discarded)</span>{/if}
  {#if $pipeline?.layers[0]?.bypassed && $blockIdx > 0}<span class="bypass-label">Layer 1 remains bypassed · shown in the orange dashed background card</span>{/if}
  {#if maskedHead}<span class="head-mask-label">Head {$attentionHeadIdx + 1} output zeroed · attention is still computed</span>{/if}
  {#if $layer?.masked_weights}<button class="mlp-mask-label" on:click={() => inspect('mlp')}>{$layer.masked_weights.toLocaleString()} / {$layer.total_mlp_weights.toLocaleString()} MLP parameters masked</button>{/if}
  {#if !$layer?.bypassed && !$layer?.masked_weights && !maskedHead}<span>Selected layer/head: all pathways retained</span>{/if}
</div>
{#if weights}
  <button class="weight-sample" aria-label="Inspect measured MLP weight masks" on:click={() => inspect('mlp')}>
    <span>MLP weight sample · coral = masked</span>
    <svg viewBox="0 0 144 25" role="img" aria-label="Measured MLP weight mask sample">
      {#each weights.sample_values as value, i}
        <rect x={(i % 24) * 6} y={Math.floor(i / 24) * 5} width="5" height="4"
          fill={weights.sample?.[i] ? '#d98e78' : '#c7c2e4'}><title>{weights.sample?.[i] ? 'Masked' : 'Retained'}: {value}</title></rect>
      {/each}
    </svg>
  </button>
{/if}
