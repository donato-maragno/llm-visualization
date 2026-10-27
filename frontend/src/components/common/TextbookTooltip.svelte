<script lang="ts">
	import {
		isTextbookOpen,
		textbookCurrentPage,
		textbookCurrentPageId,
		textbookPreviousPage,
		userId
	} from '~/store';
	import { textPages } from '~/utils/textbookPages';
	import { inspect } from '../../bridge';

	export let id: string;

	function openTextbook(e) {
		e.stopPropagation();
		e.preventDefault();
		inspect(['embedding','token-embedding','positional-encoding'].includes(id) ? 'embeddings' : ['qkv'].includes(id) ? 'qkv' : ['self-attention','masked-self-attention','multi-head','output-concatenation'].includes(id) ? 'attention' : ['blocks'].includes(id) ? 'blocks' : ['output-probabilities','temperature','sampling'].includes(id) ? 'probabilities' : 'mlp');

		const pageIndex = textPages.findIndex((page) => page.id === id);
		if (pageIndex !== -1) {
			textbookPreviousPage.set($textbookCurrentPage);
			isTextbookOpen.set(true);
			textbookCurrentPage.set(pageIndex);
			textbookCurrentPageId.set(id);
		}

		window.dataLayer?.push({
			event: `open-textbook`,
			page_id: id,
			open_via: 'tooltip',
			user_id: $userId
		});
	}
</script>

<div
	{id}
	class="textbook-tooltip"
	data-click={`textbook-tooltip`}
	on:click={openTextbook}
	role="button"
	tabindex="0"
	on:keydown={(e) => {
		if (e.key === 'Enter' || e.key === ' ') {
			openTextbook(e);
		}
	}}
>
	<slot></slot>
</div>

<style lang="scss">
	.textbook-tooltip {
		cursor: help;
	}
</style>
