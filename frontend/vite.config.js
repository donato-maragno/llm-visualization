import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  base: '/static/transformer/',
  plugins: [svelte({preprocess: vitePreprocess()})],
  resolve: {alias: {'~': fileURLToPath(new URL('./src', import.meta.url))}},
  css: {preprocessorOptions: {scss: {
    additionalData: `@use "${fileURLToPath(new URL('./src/styles/variables.scss', import.meta.url)).replaceAll('\\', '/')}" as *;`,
    silenceDeprecations: ['legacy-js-api', 'import', 'global-builtin', 'color-functions']
  }}},
  build: {
    outDir: '../web/transformer',
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {output: {
      entryFileNames: 'explainer.js',
      assetFileNames: asset => asset.name?.endsWith('.css') ? 'explainer.css' : 'assets/[name]-[hash][extname]'
    }}
  }
});
