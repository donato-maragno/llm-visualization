import { inspect } from '../bridge';

// No analytics in the local application.
export const ga = () => {};
export function onClickReadMore(event: Event, id: string) {
  event.stopPropagation();
  inspect(id.includes('attention') ? 'attention' : id.includes('embedding') ? 'embeddings' : 'mlp');
}
