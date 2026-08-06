<script lang="ts">
	/**
	 * The area list, as a file tree — because that is what the areas are:
	 * one `labels/<aoi>.geojson` per row, grouped into the three jobs
	 * (train / validation / not staged) rather than eighteen flat rows.
	 */
	export type AoiInfo = {
		name: string;
		role: string;
		split: string;
		notes: string;
		extent: [number, number, number, number];
		staged: boolean;
		total: number;
		reviewed: number;
	};

	type Group = { label: string; items: AoiInfo[] };

	let {
		groups,
		current,
		loading,
		loadError,
		onopen,
	}: {
		groups: Group[];
		current: AoiInfo | null;
		loading: boolean;
		loadError: string;
		onopen: (aoi: AoiInfo) => void;
	} = $props();

	/** Folder open/closed per group; "Not staged" starts closed — it is noise. */
	let collapsed = $state<Record<string, boolean>>({ 'Not staged': true });

	function tally(items: AoiInfo[]): { reviewed: number; total: number } {
		return items.reduce(
			(acc, a) => ({ reviewed: acc.reviewed + a.reviewed, total: acc.total + a.total }),
			{ reviewed: 0, total: 0 },
		);
	}
</script>

{#snippet fileIcon()}
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		stroke-width="1.8"
		stroke-linecap="round"
		stroke-linejoin="round"
		class="size-[13px] shrink-0 text-dim"
		aria-hidden="true"
	>
		<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
		<path d="M14 3v5h5" />
	</svg>
{/snippet}

{#snippet folderIcon()}
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		stroke-width="1.8"
		stroke-linecap="round"
		stroke-linejoin="round"
		class="size-[13px] shrink-0 text-dim"
		aria-hidden="true"
	>
		<path d="M3 7a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
	</svg>
{/snippet}

<nav class="min-h-0 flex-1 overflow-y-auto px-2 py-2.5" aria-label="Areas">
	{#if loadError}
		<p
			class="m-1 rounded border border-reject/50 bg-reject/10 px-2.5 py-2 text-xs text-reject"
			role="alert"
		>
			Could not load areas — {loadError}. Is the dev server running?
		</p>
	{/if}
	{#if loading}
		<div class="grid gap-1.5 px-1 pt-1" aria-hidden="true">
			{#each Array(10) as _, i (i)}
				<div class="h-[26px] animate-pulse rounded bg-raised" style="opacity:{1 - i * 0.07}"></div>
			{/each}
		</div>
	{/if}
	{#each groups as group (group.label)}
		{@const groupTally = tally(group.items)}
		{@const isCollapsed = collapsed[group.label] ?? false}
		<div class="pt-1 first:pt-0">
			<button
				onclick={() => (collapsed[group.label] = !isCollapsed)}
				aria-expanded={!isCollapsed}
				class="flex w-full items-center gap-1.5 rounded px-2 py-[6px] text-left text-[13px] font-medium text-muted transition-colors duration-150 hover:text-fg"
			>
				<svg
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
					class="size-3 shrink-0 text-dim transition-transform duration-150 {isCollapsed
						? '-rotate-90'
						: ''}"
					aria-hidden="true"
				>
					<path d="m6 9 6 6 6-6" />
				</svg>
				{@render folderIcon()}
				{group.label}
				<span class="ml-auto font-mono text-[11px] font-normal text-dim"
					>{groupTally.reviewed}/{groupTally.total}</span
				>
			</button>
			{#if !isCollapsed}
				<div class="mt-0.5 mb-1 ml-[19px] grid gap-px">
					{#each group.items as aoi (aoi.name)}
						{@const done = aoi.total > 0 && aoi.reviewed === aoi.total}
						{@const pct = aoi.total ? (aoi.reviewed / aoi.total) * 100 : 0}
						<button
							onclick={() => onopen(aoi)}
							disabled={!aoi.staged}
							aria-current={current?.name === aoi.name}
							class="flex w-full items-center gap-1.5 rounded px-2 py-[5px] text-left transition-[background-color,color,transform] duration-150 ease-[var(--ease-out)]
								hover:not-disabled:bg-raised hover:not-disabled:text-fg
								active:not-disabled:scale-[0.985]
								focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent
								disabled:cursor-default disabled:opacity-35
								aria-[current=true]:bg-raised aria-[current=true]:text-fg
								{current?.name === aoi.name ? 'text-fg' : 'text-muted'}"
						>
							{@render fileIcon()}
							<span class="truncate text-[13px]">{aoi.name}.geojson</span>
							{#if aoi.staged}
								<span class="ml-auto flex shrink-0 items-center gap-2">
									<span class="h-[5px] w-9 overflow-hidden rounded-full bg-line">
										<span
											class="block h-full rounded-full bg-truck transition-[width] duration-200 ease-[var(--ease-out)]"
											style="width:{pct}%"
										></span>
									</span>
									<span class="font-mono text-[11px] {done ? 'text-truck' : 'text-dim'}">
										{aoi.reviewed}/{aoi.total}
									</span>
								</span>
							{:else}
								<span
									class="ml-auto shrink-0 rounded-[3px] border border-line px-1 py-px font-mono text-[9px] tracking-[0.08em] text-dim uppercase"
									>{aoi.role}</span
								>
							{/if}
						</button>
					{/each}
				</div>
			{/if}
		</div>
	{/each}
</nav>
