<script>
  import { pipeline, blockIdx } from './store';
  import { notify } from './bridge';

  $: showFirstLayer = $pipeline?.layers[0]?.bypassed && $blockIdx > 0;
</script>

{#if showFirstLayer}
  <button type="button" class="bypassed-layer-context" data-bypassed-layer="1"
    aria-label={`Inspect bypassed layer 1; currently inspecting layer ${$blockIdx + 1}`}
    on:click={() => notify('select-layer', {layer: 0})}>
    <svg viewBox="0 0 236 148" role="img" aria-label="Bypassed layer 1 behind the active transformer layer">
      <title>Layer context only, separate from the attention-head stack. Click to return to layer 1.</title>
      <g class="background-layer">
        <path class="background-depth" d="M30 22 L42 12 H222 V106 L210 116 M210 22 L222 12 M210 116 V22 H30 V116 H210" />
        <text x="48" y="35">Layer 1 · bypassed</text>
        <path class="background-symbols" d="M50 49 h28 m9 0 h32 m9 0 h30 m9 0 h30" />
      </g>
      <g class="active-context-layer">
        <path class="active-depth" d="M10 64 L22 54 H202 V132 L190 142 H10 Z M190 64 L202 54 M190 64 V142 M10 64 H190" />
        <text class="active-context-title" x="24" y="83">Layer {$blockIdx + 1} · inspecting</text>
        <g class="context-architecture">
          <rect x="25" y="100" width="29" height="20" rx="3" />
          <rect x="72" y="100" width="60" height="20" rx="3" />
          <rect x="150" y="100" width="28" height="20" rx="3" />
          <path d="M54 110 H72 M132 110 H150" />
          <text x="39" y="113" text-anchor="middle">QKV</text>
          <text x="102" y="113" text-anchor="middle">Attention</text>
          <text x="164" y="113" text-anchor="middle">MLP</text>
        </g>
      </g>
    </svg>
    <span>Layer 1 stays bypassed · click to inspect</span>
  </button>
{/if}

<style>
  .bypassed-layer-context {position:absolute;right:22px;bottom:12px;width:236px;padding:0;border:0;background:transparent;text-align:left;z-index:600;cursor:pointer}
  svg {display:block;width:100%;height:148px;overflow:visible}
  text {font:11px ui-sans-serif,system-ui,sans-serif}
  .background-layer {fill:none;stroke:#e49435;stroke-width:1.7;stroke-dasharray:7 4;stroke-linejoin:round}
  .background-layer text {fill:#c47d27;stroke:none;font-weight:550}
  .background-symbols {opacity:.28}
  .active-depth {fill:rgba(255,255,255,.92);stroke:#cbd4df;stroke-width:1;stroke-linejoin:round}
  .active-context-title {fill:#6d7e92}
  .context-architecture rect {fill:none;stroke:#d1d9e4}
  .context-architecture path {fill:none;stroke:#ccd6e3}
  .context-architecture text {fill:#98a4b4;font-size:8px}
  span {display:block;margin:3px 0 0 10px;color:#b7813c;font-size:9px}
  button:hover .background-depth {stroke:#ce7e22;stroke-width:2.1}
</style>
