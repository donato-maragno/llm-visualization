import { mount } from 'svelte';
import App from './App.svelte';
import { notify, receive } from './bridge';
import './styles/app.css';
import './styles/layout.scss';
import 'katex/dist/katex.min.css';

const app = mount(App, {target: document.getElementById('app')});
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== parent || event.data?.type !== 'llm-visualizer:update') return;
  receive(event.data.payload);
});
notify('ready');
