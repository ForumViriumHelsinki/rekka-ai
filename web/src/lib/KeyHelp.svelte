<script lang="ts">
	/**
	 * The keyboard reference, rendered from the same table that handles the
	 * keys. Two lists of the same bindings is how a help panel starts lying.
	 */
	let { bindings }: { bindings: { keys: string; label: string }[] } = $props();
</script>

<details class="group border-t border-line">
	<summary
		class="cursor-pointer list-none px-3.5 py-2.5 text-[10px] font-semibold tracking-[0.08em] text-dim uppercase select-none hover:text-muted"
	>
		Keys
		<span class="float-right transition-transform duration-150 group-open:rotate-90">›</span>
	</summary>
	<dl class="grid grid-cols-[auto_1fr] gap-x-2.5 gap-y-1.5 px-3.5 pb-3.5 text-xs">
		{#each bindings as { keys, label } (label)}
			<dt class="flex gap-1">
				{#each keys.split(' ') as k (k)}
					<kbd
						class="min-w-[17px] rounded border border-b-2 border-line bg-ink px-1 text-center font-mono text-[10px] leading-normal text-muted"
						>{k}</kbd
					>
				{/each}
			</dt>
			<dd class="text-muted">{label}</dd>
		{/each}
	</dl>
</details>
