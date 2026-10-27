<script>
  import { onMount, tick, afterUpdate } from 'svelte';
  import { pipeline, tokens, expandedBlock, blockIdx, attentionHeadIdx, layer, vectorHeight, headContentHeight, diagramWidth } from './store';
  import { navigate, notify } from './bridge';
  import Sankey from './components/Sankey.svelte';
  import Embedding from './components/Embedding.svelte';
  import QKV from './components/QKV.svelte';
  import Attention from './components/Attention.svelte';
  import Mlp from './components/Mlp.svelte';
  import SubsequentBlocks from './components/SubsequentBlocks.svelte';
  import LinearSoftmax from './components/LinearSoftmax.svelte';
  import LayerNavigation from './LayerNavigation.svelte';
  import PruningOverlay from './PruningOverlay.svelte';

  $: count = $tokens.length || 1;
  $: layoutHeight = Math.max(740, count * 48 + 180);
  $: height = layoutHeight + 140;
  $: $vectorHeight = Math.min(22, Math.max(12, (layoutHeight - 150 - (count - 1) * 8) / count / 3.4));
  $: $headContentHeight = Math.max(count * $vectorHeight * 3 + (count - 1) * 8, 320);
  $: maskedHead = $layer?.masked_heads.includes($attentionHeadIdx);
  let container;
  let observer;
  const observed = new WeakSet();
  const report = () => {
    // Ignore offscreen, hidden tooltip nodes when measuring expanded content.
    const panels = [...document.querySelectorAll('.step > .content,.head-content,.embedding .token-column,.softmax .content-row')];
    const width = Math.ceil(Math.max($diagramWidth, ...panels.map(node => node.getBoundingClientRect().right)));
    notify('size', {height, width});
  };
  let wheelDelta = 0, lastWheel = 0, lastStep = -Infinity;
  function scrollLayer(event) {
    if (!event.target.closest('.layer-navigation') || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY) || !event.deltaY) return;
    event.preventDefault();
    const now = performance.now();
    if (now - lastWheel > 250 || Math.sign(wheelDelta) !== Math.sign(event.deltaY)) wheelDelta = 0;
    lastWheel = now;
    wheelDelta += event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
    if (Math.abs(wheelDelta) < 24 || now - lastStep < 120) return;
    navigate('layer', Math.sign(wheelDelta)); wheelDelta = 0; lastStep = now;
  }
  onMount(() => {
    observer = new ResizeObserver(report);
    observer.observe(container);
    container.addEventListener('wheel', scrollLayer, {passive:false});
    return () => { observer.disconnect(); container.removeEventListener('wheel', scrollLayer); };
  });
  afterUpdate(() => {
    if (!observer) return;
    document.querySelectorAll('.resize-watch').forEach(node => {
      if (!observed.has(node)) { observer.observe(node); observed.add(node); }
    });
    report();
  });
</script>

<div bind:this={container} class="main-section" class:has-data={!!$pipeline}
  class:block-bypassed={$layer?.bypassed} class:head-pruned={maskedHead}
  style={`--vector-height:${$vectorHeight}px;--title-height:104px;--content-height:${layoutHeight - 104}px;height:${height}px;width:${$diagramWidth}px`}>
  {#if $pipeline}
    {#if $expandedBlock.id}
      <button class="close-expansion" on:click={() => expandedBlock.set({id:null})}>Close expanded view ×</button>
    {/if}
    <div class="sankey" class:attention={$expandedBlock.id === 'attention'}><Sankey /></div>
    <div class="nodes resize-watch">
      <div class="steps" class:expanded={!!$expandedBlock.id}>
        <Embedding className="step" />
        <div class="blocks relative">
          <div class="block-navigation">
            <LayerNavigation name="QKV" id="qkv" />
            <LayerNavigation name="Self Attention" id="attention" />
            <LayerNavigation name="MLP" id="mlp" />
          </div>
          <div class="block-steps main" class:initial={$blockIdx === 0}>
            <QKV className="step" /><Attention className="step" /><Mlp className="step" />
          </div>
          <div class="transition-watch" aria-hidden="true"></div>
        </div>
        <SubsequentBlocks className="step" />
        <LinearSoftmax className="step" />
      </div>
    </div>
    <PruningOverlay />
  {:else}
    <p class="waiting">Waiting for the measured model snapshot…</p>
  {/if}
</div>
