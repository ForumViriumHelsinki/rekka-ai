<script lang="ts">
	/**
	 * The overlay above the map: which class the next verdict applies, the
	 * drawing width while drawing, and whether the work has reached disk.
	 *
	 * A toolbar as well as keys, because "which class am I about to press"
	 * should be visible rather than remembered — and the save badge sits here
	 * because it is the one piece of state the operator must never have to go
	 * looking for.
	 */
	import { COLOURS } from '$lib/mapStyles';
	import { CLASSES, type Klass } from '$lib/schema';

	let {
		activeClass,
		drawing,
		width,
		hasSelection,
		saveState,
		saveError,
		onclassify,
		ondelete,
	}: {
		activeClass: Klass;
		drawing: boolean;
		width: number;
		hasSelection: boolean;
		saveState: string;
		saveError: string;
		onclassify: (klass: Klass) => void;
		ondelete: () => void;
	} = $props();
</script>

<div class="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-2">
	<div
		class="pointer-events-auto flex items-center gap-0.5 rounded-lg border border-line bg-ink/85 p-1 shadow-lg shadow-black/25 backdrop-blur-md"
	>
		{#each CLASSES as klass (klass)}
			{@const active = activeClass === klass}
			<button
				onclick={() => onclassify(klass)}
				aria-pressed={active}
				title="classify as {klass} ({klass[0]})"
				class="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] transition-colors duration-150
					focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent
					{active
					? 'bg-raised text-fg shadow-[inset_0_0_0_1px_var(--color-line)]'
					: 'text-muted hover:text-fg'}"
			>
				<i class="size-[7px] rounded-full" style="background:{COLOURS[klass]}"></i>
				{klass}
				<kbd class="font-mono text-[9px] text-dim uppercase">{klass[0]}</kbd>
			</button>
		{/each}
		{#if drawing}
			<span class="mr-1 ml-1 border-l border-line pl-2 font-mono text-[11px] text-accent">
				{width.toFixed(1)} m
			</span>
		{:else if hasSelection}
			<button
				onclick={ondelete}
				title="delete the selected box entirely (Del)"
				class="mr-0.5 ml-1 flex items-center gap-1.5 rounded border-l border-line px-2 py-1 text-[11px] text-reject transition-colors duration-150 hover:bg-reject/15
					focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-reject"
			>
				<svg
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					class="size-3"
					aria-hidden="true"
				>
					<path
						d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"
					/>
				</svg>
				delete
			</button>
		{/if}
	</div>
	<span
		class="pointer-events-auto rounded-md border px-2 py-1 font-mono text-[10px] tracking-[0.1em] uppercase backdrop-blur-md transition-colors duration-150
			{saveState === 'SAVE FAILED'
			? 'border-reject/50 bg-reject/15 text-reject'
			: saveState === 'unsaved'
				? 'border-accent/40 bg-ink/85 text-accent'
				: 'border-line bg-ink/85 text-dim'}"
		title={saveError || undefined}>{saveState}</span
	>
</div>
