<script lang="ts">
  // Upstream vector texture, changed to draw measured samples only.
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import tailwindConfig from '../../../tailwind.config';
  import resolveConfig from 'tailwindcss/resolveConfig';
  import { sharedRanges } from '~/store';

  const { theme } = resolveConfig(tailwindConfig);
  export let active = false;
  export let data: number[] = [];
  export let kind = 'unrecorded';
  export let colorScale: string | ((t: number, i: number) => string) = 'gray';
  let canvas: HTMLCanvasElement;
  let scale = 1;

  function drawCanvas() {
    if (!canvas) return;
    const width = canvas.parentElement.clientWidth, height = canvas.parentElement.clientHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = Math.max(1, width * 2);
    canvas.height = Math.max(1, height * 2);
    if (!data.length) return;
    const tone = typeof colorScale === 'string' ? theme.colors[colorScale]?.[400] || '#9ca3af' : null;
    data.forEach((value, i) => {
      const magnitude = Math.min(1, Math.abs(value) / scale);
      ctx.fillStyle = typeof colorScale === 'function'
        ? colorScale(0.5 + value / (2 * scale), i)
        : d3.interpolate('white', value < 0 ? '#a78bfa' : tone)(magnitude);
      ctx.fillRect(0, i * canvas.height / data.length, canvas.width, Math.ceil(canvas.height / data.length));
    });
  }
  $: scale = $sharedRanges[kind] || Math.max(0.000001, ...data.map(Math.abs));
  $: if (canvas && data && scale && colorScale) drawCanvas();
  onMount(() => {
    const observer = new ResizeObserver(drawCanvas);
    observer.observe(canvas.parentElement);
    return () => observer.disconnect();
  });
</script>

<canvas bind:this={canvas} class:active data-field={kind} data-sample={JSON.stringify(data)}
  title={data.length ? `Measured ${kind} sample: ${data.join(', ')}` : 'Architectural vector; this intermediate was not recorded.'}
  aria-label={data.length ? `Measured ${kind} vector sample` : 'Unrecorded architectural vector'}></canvas>

<style>
  canvas {position:absolute;inset:0;opacity:0;display:block;width:100%;height:100%;transition:opacity .2s}
  canvas.active,canvas:hover {opacity:1}
</style>
