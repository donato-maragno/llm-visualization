<script>
  import { onMount, tick } from 'svelte';
  import { pipeline, layer, blockIdx, attentionHeadIdx, topology } from './store';
  import { inspect } from './bridge';
  let paths = [];
  let svg;
  $: currentTopology = $topology?.layers?.[$blockIdx];
  $: weights = currentTopology?.mlp_weights?.find(weight => weight.name.endsWith('c_fc.weight'));
  $: maskedHead = $layer?.masked_heads.includes($attentionHeadIdx);
  async function redraw() {
    await tick();
    if (!$layer?.bypassed || !svg) { paths = []; return; }
    const starts = [...document.querySelectorAll('.block-steps.main .qkv .embedding-column .vector')];
    const ends = [...document.querySelectorAll('.block-steps.main .mlp .mlp-out-column .vector')];
    const rect = svg.getBoundingClientRect();
    const floor = document.querySelector('.head-content')?.getBoundingClientRect().bottom - rect.top + 8 * ($pipeline.head_count - 1) + 28;
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
    return () => observer.disconnect();
  });
</script>

<svg class="pruning-overlay" bind:this={svg} aria-label="Applied pruning overlays">
  {#each paths as path}<path class="bypass-route" d={path} />{/each}
</svg>
<div class="pruning-summary" aria-live="polite">
  {#if $layer?.bypassed}<span class="bypass-label">Block bypassed · effective output = input (internal computation discarded)</span>{/if}
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
