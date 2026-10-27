<script>
  import { blockIdx, modelMeta } from './store';
  import { navigate } from './bridge';
  export let name;
  export let id;
</script>

<div class="layer-navigation" data-nav={id} aria-label={`Layer navigation for ${name}`}>
  <button aria-label={`Previous layer in ${name}`} disabled={$blockIdx === 0} on:click={() => navigate('layer', -1)}>‹</button>
  <span>Layer {$blockIdx + 1} / {$modelMeta.layer_num}</span>
  <button aria-label={`Next layer in ${name}`} disabled={$blockIdx === $modelMeta.layer_num - 1} on:click={() => navigate('layer', 1)}>›</button>
</div>
