<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import Map from 'ol/Map';
	import View from 'ol/View';
	import VectorLayer from 'ol/layer/Vector';
	import VectorSource from 'ol/source/Vector';
	import Select from 'ol/interaction/Select';
	import Translate from 'ol/interaction/Translate';
	import MouseWheelZoom from 'ol/interaction/MouseWheelZoom';
	import { defaults as defaultInteractions } from 'ol/interaction/defaults';
	import ScaleLine from 'ol/control/ScaleLine';
	import Zoom from 'ol/control/Zoom';
	import Attribution from 'ol/control/Attribution';
	import { Fill, Stroke, Style, Text } from 'ol/style';
	import Polygon from 'ol/geom/Polygon';
	import { fromExtent } from 'ol/geom/Polygon';
	import Feature from 'ol/Feature';
	import type { FeatureLike } from 'ol/Feature';
	import { containsCoordinate } from 'ol/extent';
	import { click } from 'ol/events/condition';
	// Without this the zoom, attribution and scale controls render unstyled.
	import 'ol/ol.css';

	import { GRID, orthoLayer, resolutionAt, MAX_ZOOM } from '$lib/grid';
	import {
		boxFromCentreline,
		centrelineOf,
		clampWidth,
		measure,
		scaleCentreline,
		rotateCentreline,
		DEFAULT_WIDTH_M,
		WIDTH_STEP_M,
		LENGTH_STEP_M,
		LENGTH_COARSE_M,
		ROTATE_STEP_DEG,
		ROTATE_COARSE_DEG,
		type Coord,
	} from '$lib/obb';
	import { CLASSES, LATEST_LAYER, SOURCE_ZOOM, type Klass } from '$lib/schema';
	import AreaTree, { type AoiInfo } from '$lib/AreaTree.svelte';
	import KeyHelp from '$lib/KeyHelp.svelte';
	import ClassToolbar from '$lib/ClassToolbar.svelte';
	import { LabelStore, refreshMeasurements } from '$lib/labelStore.svelte';
	import { UndoStack } from '$lib/undo.svelte';
	import { EndHandles } from '$lib/endHandles';
	import { DrawMode } from '$lib/drawMode.svelte';
	import {
		COLOURS,
		aoiStyle,
		handleStyle,
		neighbourStyle,
		selectedStyleFor,
		sketchStyle,
		styleFor,
	} from '$lib/mapStyles';

	let aois = $state<AoiInfo[]>([]);
	let loading = $state(true);
	let current = $state<AoiInfo | null>(null);
	let width = $state(DEFAULT_WIDTH_M);
	let activeClass = $state<Klass>('truck');
	let selectedInfo = $state('');
	let loadError = $state('');
	/** Set from /api/aois before the map is built; the initial value is only
	 * a fallback for a failed fetch. */
	let layerName = $state(LATEST_LAYER);

	/** Areas grouped so the list reads as three jobs, not eighteen rows. */
	const groups = $derived(
		[
			{
				label: 'Train',
				items: aois.filter((a) => a.staged && a.split === 'train' && a.role === 'positive'),
			},
			{ label: 'Validation', items: aois.filter((a) => a.staged && a.split === 'validation') },
			{ label: 'Not staged', items: aois.filter((a) => !a.staged) },
		].filter((g) => g.items.length),
	);

	const overall = $derived(
		aois.reduce((acc, a) => ({ reviewed: acc.reviewed + a.reviewed, total: acc.total + a.total }), {
			reviewed: 0,
			total: 0,
		}),
	);

	let map: Map;
	let select: Select;
	let translate: Translate;
	let wheelZoom: MouseWheelZoom;
	/** Hand-drawing a new box — see drawMode.svelte.ts. Created with the map;
	 * `drawing`/`phase` live there because the footer and keymap read them. */
	let drawMode = $state<DrawMode | null>(null);
	/** The current area's bbox, drawn so "am I still inside the area" is
	 * visible rather than memorised. */
	let aoiSource: VectorSource;
	/** Every *other* area's bbox — where else there is work, and a click target
	 * to go there. Kept apart from `aoiSource` so the current boundary keeps its
	 * own styling, and so selection can stay scoped away from both. */
	let neighbourSource: VectorSource;
	/** Draggable ends of the selected box — the one-ended length edit. Created
	 * with the map; every box-moving code path pokes `endHandles.sync()`. */
	let endHandles: EndHandles | null = null;
	/** Held so onDestroy can remove it; see the note at its registration. */
	let onWheel: ((event: WheelEvent) => void) | null = null;
	/** Owns the features, the dirty flag and the save path — see labelStore. */
	const store = new LabelStore({
		onCounts: (next) => {
			// Mirror into the sidebar row so the tree shows progress without the
			// store knowing a tree exists.
			const entry = aois.find((a) => a.name === store.area);
			if (entry) {
				entry.reviewed = next.reviewed;
				entry.total = next.total;
			}
		},
	});
	/** One step back for a change the operator did not mean. Autosave commits
	 * every edit within a second, so this is the only in-session safety net. */
	const undoStack = new UndoStack();
	const source = store.source;
	const markDirty = () => store.markDirty();
	const save = () => store.save();

	/** The one source of truth for "which feature is selected". Click and the
	 * N/P keys both end here, so the highlight cannot diverge between them.
	 * $state.raw: a Feature must keep its identity for layerStyle's === check,
	 * so it must not be wrapped in a reactive proxy. */
	let selected = $state.raw<Feature | null>(null);

	function setSelected(feature: Feature | null) {
		if (selected === feature) return;
		selected = feature;
		describe(feature ?? undefined);
		endHandles?.sync();
		source?.changed(); // layerStyle closes over `selected`
	}

	function layerStyle(feature: FeatureLike): Style | Style[] {
		const f = feature as Feature;
		return f === selected ? selectedStyleFor(f) : styleFor(f);
	}

	async function open(aoi: AoiInfo) {
		if (store.dirty) await store.save();
		current = aoi;
		drawMode?.stop();
		select.getFeatures().clear();
		setSelected(null);
		selectedInfo = '';
		try {
			await store.load(aoi.name);
		} catch (e) {
			loadError = e instanceof Error ? e.message : String(e);
			return;
		}
		loadError = '';
		undoStack.clear();
		aoiSource.clear();
		aoiSource.addFeature(new Feature(fromExtent(aoi.extent)));
		showNeighbours();
		// minResolution, not maxResolution: OL's FitOptions has no such key and
		// silently ignored it, so opening a small area used to zoom straight
		// past the imagery's real detail. "Minimum resolution we zoom to" is the
		// floor on metres-per-pixel — z16 is where the labels were drawn.
		map.getView().fit(aoi.extent, { padding: [24, 24, 24, 24], minResolution: resolutionAt(16) });
	}

	/** Redraw the other areas' boundaries. Called on every open, and once before
	 * the first, when nothing is open and all of them are a way in. */
	function showNeighbours() {
		neighbourSource.clear();
		for (const aoi of aois) {
			if (aoi.name === current?.name) continue;
			const feature = new Feature(fromExtent(aoi.extent));
			feature.set('name', aoi.name);
			neighbourSource.addFeature(feature);
		}
	}

	/** Turn the axis DrawMode committed into a real feature. DrawMode owns the
	 * gesture; the page owns what a box means: source, undo, selection, dirty. */
	function commitBox(nose: Coord, tail: Coord) {
		const feature = new Feature(new Polygon([boxFromCentreline(nose, tail, width)]));
		feature.setProperties({
			class: activeClass,
			status: 'added',
			confidence: null,
			aoi: current?.name ?? '',
			source_layer: layerName,
			zoom: SOURCE_ZOOM,
		});
		refreshMeasurements(feature);
		source.addFeature(feature);
		undoStack.added(feature);
		// Select what was just drawn. A box is at its most wrong the moment it
		// is placed, so the next action is almost always to correct it — making
		// that cost a click back onto a box you are already looking at was the
		// confusing part of the draw flow.
		select.getFeatures().clear();
		select.getFeatures().push(feature);
		setSelected(feature);
		markDirty();
	}

	function applyClass(klass: Klass) {
		activeClass = klass;
		const picked = select.getFeatures().getArray();
		for (const feature of picked) {
			undoStack.edit(feature);
			feature.set('class', klass);
			if ((feature.get('status') ?? 'candidate') !== 'added') feature.set('status', 'confirmed');
		}
		if (picked.length) {
			// The box stays selected and highlighted: its class colour reads
			// through the halo now, so there is nothing to reveal by dropping
			// it — and dropping it used to leave the box uneditable until it was
			// clicked again, since every edit is guarded on `selected`.
			setSelected(picked[0]);
			describe(picked[0]);
			source.changed(); // the class colour changed underneath the halo
			markDirty();
		}
	}

	function rejectSelected() {
		const picked = select.getFeatures().getArray();
		for (const feature of picked) {
			undoStack.edit(feature);
			feature.set('status', 'rejected');
			feature.set('class', '');
		}
		if (picked.length) {
			setSelected(picked[0]);
			describe(picked[0]);
			source.changed(); // rejected boxes restyle to a dashed grey
			markDirty();
		}
	}

	/** Step back, and put the selection on what came back so the change is
	 * visible rather than merely reversed. Saves like any other edit: the file
	 * already holds the mistake, so undo is only undone once it reaches disk. */
	function undoLast() {
		// Checked before, not after: undoing a hand-drawn box also returns null,
		// and treating that as "nothing happened" would skip the save.
		if (!undoStack.depth) return;
		const restored = undoStack.undo(source);
		select.getFeatures().clear();
		if (restored) {
			select.getFeatures().push(restored);
			setSelected(restored);
			describe(restored);
		} else {
			setSelected(null);
			selectedInfo = '';
		}
		// Undoing a geometry edit restores the ring on the *same* feature, so
		// setSelected above may early-return with the handles left stale.
		endHandles?.sync();
		source.changed();
		markDirty();
	}

	function deleteSelected() {
		for (const feature of [...select.getFeatures().getArray()]) {
			undoStack.removed(feature, source.getFeatures().indexOf(feature));
			source.removeFeature(feature);
		}
		select.getFeatures().clear();
		setSelected(null);
		markDirty();
	}

	/** Shift+scroll on a selected box rebuilds it around its own axis with a
	 * new width — the modifier keeps plain scroll free for zooming during
	 * N/P review. */
	function resizeSelected(delta: number) {
		if (!selected) return;
		const geometry = selected.getGeometry() as Polygon;
		const ring = geometry.getCoordinates()[0] as Coord[];
		const m = measure(ring);
		const next = clampWidth(m.width + delta);
		if (next === m.width) return;
		undoStack.edit(selected);
		const [nose, tail] = centrelineOf(ring);
		geometry.setCoordinates([boxFromCentreline(nose, tail, next)]);
		width = next; // the next drawn box starts from what was just corrected
		refreshMeasurements(selected);
		describe(selected);
		endHandles?.sync();
		markDirty();
	}

	/** Rebuild the selected box from a corrected centreline, keeping its width.
	 * Every shape edit goes through here rather than through the ring, so the
	 * box stays a rectangle by construction — see the note in obb.ts. */
	function reshapeSelected(correct: (nose: Coord, tail: Coord, width: number) => [Coord, Coord]) {
		if (!selected) return;
		const geometry = selected.getGeometry() as Polygon;
		const ring = geometry.getCoordinates()[0] as Coord[];
		undoStack.edit(selected);
		const [nose, tail] = centrelineOf(ring);
		const boxWidth = measure(ring).width;
		const [nextNose, nextTail] = correct(nose, tail, boxWidth);
		geometry.setCoordinates([boxFromCentreline(nextNose, nextTail, boxWidth)]);
		refreshMeasurements(selected);
		describe(selected);
		endHandles?.sync();
		markDirty();
	}

	/** Jump to the next feature still awaiting a verdict. Traversal follows a
	 * sweep — northernmost row first, west to east within a row — so N always
	 * lands near the previous box instead of teleporting in file order. */
	function nextCandidate(step = 1) {
		const ROW_M = 100; // candidates within a band this tall count as one row
		const centre = (f: Feature): Coord => {
			const [minX, minY, maxX, maxY] = (f.getGeometry() as Polygon).getExtent();
			return [(minX + maxX) / 2, (minY + maxY) / 2];
		};
		const sweep = (list: Feature[]) =>
			list.sort((a, b) => {
				const [ax, ay] = centre(a);
				const [bx, by] = centre(b);
				const row = Math.round(by / ROW_M) - Math.round(ay / ROW_M); // north first
				return row !== 0 ? row : ax - bx; // then west first
			});
		// Walk the sweep over *every* feature, skipping the ones we do not want,
		// rather than over a filtered ring. The filtered ring drops the current
		// box the moment it gets a verdict, and stepping from a box that is not
		// in the ring lands at an arbitrary offset — N happened to look right
		// while P jumped to the second-to-last box in the area.
		const ring = sweep(source.getFeatures().slice());
		if (!ring.length) {
			selectedInfo = 'no features here';
			return;
		}
		// Unreviewed candidates come first while any remain. Once an area is
		// fully reviewed N/P must still move: revisiting a verdict is the whole
		// of a correction round, and going dead at 100% strands the operator on
		// whichever box they last touched.
		const unreviewed = (f: Feature) => (f.get('status') ?? 'candidate') === 'candidate';
		const wanted = ring.some(unreviewed) ? unreviewed : () => true;
		const mod = (i: number) => ((i % ring.length) + ring.length) % ring.length;
		const currentFeature = select.getFeatures().getArray()[0];
		// With nothing selected, N starts at the head of the sweep and P at its
		// tail — the same both-ends symmetry as stepping off either edge.
		const from = currentFeature ? ring.indexOf(currentFeature) : step > 0 ? -1 : 0;
		let index = mod(from + step);
		for (let i = 1; i < ring.length && !wanted(ring[index]); i++) index = mod(index + step);
		const target = ring[index];
		select.getFeatures().clear();
		select.getFeatures().push(target);
		// OL only dispatches the select event for pointer-driven selections,
		// not programmatic pushes — so set the highlight state directly,
		// otherwise a keyboard-first N never highlights anything.
		setSelected(target);
		// No animation: N is pressed hundreds of times a session, and a
		// duration here would make every one of them feel slow. Modest padding
		// keeps the jump zoomed close enough to judge the box by eye.
		map.getView().fit((target.getGeometry() as Polygon).getExtent(), {
			maxZoom: MAX_ZOOM,
			padding: [150, 150, 150, 150],
		});
	}

	function describe(feature: Feature | undefined) {
		if (!feature) {
			selectedInfo = '';
			return;
		}
		refreshMeasurements(feature);
		const confidence = feature.get('confidence');
		const klass = feature.get('class');
		const features = source.getFeatures();
		selectedInfo =
			`#${features.indexOf(feature) + 1}/${features.length}` +
			` · ${feature.get('status') ?? 'candidate'}` +
			`${klass ? ' · ' + klass : ''}` +
			` · ${feature.get('length_m')} × ${feature.get('width_m')} m` +
			` · ${feature.get('heading_deg')}°` +
			(confidence == null ? '' : ` · conf ${confidence}`);
	}

	/** Whether a keystroke belongs to something the operator is typing into.
	 *
	 * Broader than an input check: T/B/V/Del are single letters, so any field
	 * that swallows text — a textarea, a select, a contenteditable — would
	 * otherwise classify or delete a box while someone types in it. There is no
	 * such field on the page today; this is here so adding one is not a bug. */
	function isTyping(target: EventTarget | null): boolean {
		if (!(target instanceof HTMLElement)) return false;
		return (
			target instanceof HTMLInputElement ||
			target instanceof HTMLTextAreaElement ||
			target instanceof HTMLSelectElement ||
			target.isContentEditable
		);
	}

	/**
	 * The keymap, and the help panel, from one table.
	 *
	 * These used to be a handler and a hand-maintained list of the same
	 * bindings; adding the arrow keys meant editing both, which is the drift
	 * waiting to happen. An entry with no `on` is documentation for something
	 * the mouse does.
	 */
	type Binding = {
		keys: string;
		label: string;
		on?: string[];
		when?: (event: KeyboardEvent) => boolean;
		run?: (key: string, event: KeyboardEvent) => void;
	};

	const BINDINGS: Binding[] = [
		{
			keys: CLASSES.map((c) => c[0].toUpperCase()).join(' '),
			label: CLASSES.join(' / '),
			on: CLASSES.map((c) => c[0]),
			// Derived from CLASSES rather than spelled out, so a fourth class
			// gets its key and its help line without touching this handler.
			run: (key) => applyClass(CLASSES.find((c) => c[0] === key)!),
		},
		{ keys: 'X', label: 'reject', on: ['x'], run: () => rejectSelected() },
		{
			keys: 'N P',
			label: 'next / prev unreviewed',
			on: ['n', 'p'],
			run: (key) => nextCandidate(key === 'n' ? 1 : -1),
		},
		{
			keys: 'D',
			label: 'draw: nose, tail, scroll width',
			on: ['d'],
			run: () => (drawMode?.drawing ? drawMode.stop() : drawMode?.start()),
		},
		{
			keys: 'Enter',
			label: 'place box while drawing',
			on: ['enter'],
			when: () => drawMode?.drawing === true && drawMode?.phase === 2,
			run: () => drawMode?.commit(),
		},
		{
			keys: 'Esc',
			label: 'redo sketch / stop drawing',
			on: ['escape'],
			run: () => (drawMode?.drawing && drawMode.phase > 0 ? drawMode.cancel() : drawMode?.stop()),
		},
		{ keys: 'drag', label: 'move selected box' },
		{ keys: 'drag end', label: 'resize one end of selected box' },
		{ keys: '⇧ scroll', label: 'width of selected box' },
		{
			keys: '↑ ↓',
			label: 'length of selected box (⇧ coarse)',
			on: ['arrowup', 'arrowdown'],
			run: (key, event) => {
				const step = event.shiftKey ? LENGTH_COARSE_M : LENGTH_STEP_M;
				reshapeSelected((nose, tail, boxWidth) =>
					scaleCentreline(nose, tail, key === 'arrowup' ? step : -step, boxWidth),
				);
			},
		},
		{
			keys: '← →',
			label: 'rotate selected box (⇧ coarse)',
			on: ['arrowleft', 'arrowright'],
			run: (key, event) => {
				const turn = event.shiftKey ? ROTATE_COARSE_DEG : ROTATE_STEP_DEG;
				reshapeSelected((nose, tail) =>
					rotateCentreline(nose, tail, key === 'arrowright' ? turn : -turn),
				);
			},
		},
		{
			keys: 'Del',
			label: 'delete selected box',
			on: ['delete', 'backspace'],
			run: () => deleteSelected(),
		},
		{
			keys: '⌘/⌃ Z',
			label: 'undo the last change',
			on: ['z'],
			when: (e) => e.ctrlKey || e.metaKey,
			run: () => undoLast(),
		},
		{
			keys: '⌘/⌃ S',
			label: 'save now',
			on: ['s'],
			when: (e) => e.ctrlKey || e.metaKey,
			run: () => save(),
		},
	];

	function onKey(event: KeyboardEvent) {
		if (isTyping(event.target)) return;
		const key = event.key.toLowerCase();
		const binding = BINDINGS.find((b) => b.on?.includes(key) && (b.when?.(event) ?? true));
		if (!binding?.run) return;
		binding.run(key, event);
		event.preventDefault();
	}

	onMount(async () => {
		// Fetched before the map is built: the response carries the imagery
		// layer name, which the base layer needs at construction.
		try {
			const response = await fetch('/api/aois');
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			const data = await response.json();
			layerName = data.layer;
			aois = data.aois;
		} catch (e) {
			// A failed load used to leave an empty sidebar and no explanation,
			// while a failed save shouted. Both are the tool being unusable;
			// both should say so.
			loadError = e instanceof Error ? e.message : String(e);
			loading = false;
			return;
		}
		loading = false;

		const sketchSource = new VectorSource();
		aoiSource = new VectorSource();
		neighbourSource = new VectorSource();
		const labelLayer = new VectorLayer({ source, style: layerStyle });
		// Topmost: a handle must never be hidden by the box it belongs to, and
		// hit-testing walks top-down, so handles are found before anything else.
		const handleLayer = new VectorLayer({ source: new VectorSource(), style: handleStyle });
		// Bottom of the vector stack: a neighbour must never draw over a box
		// being judged, and hit-testing walks top-down, so labels are asked
		// first for any click.
		const neighbourLayer = new VectorLayer({ source: neighbourSource, style: neighbourStyle });
		map = new Map({
			target: 'map',
			layers: [
				orthoLayer(layerName),
				neighbourLayer,
				new VectorLayer({ source: aoiSource, style: aoiStyle }),
				labelLayer,
				new VectorLayer({ source: sketchSource, style: sketchStyle }),
				handleLayer,
			],
			// Arrows nudge the selected box's length and heading, so the map must
			// not also pan on them: two things moving at once is unreadable.
			interactions: defaultInteractions({ keyboard: false }),
			// Listed explicitly rather than defaults(): attribution is not
			// optional — the imagery is Helsinki's open data, and the scale bar
			// is how the 6 m gate gets judged by eye.
			controls: [
				new Zoom(),
				new Attribution({ collapsible: false }),
				new ScaleLine({ units: 'metric', minWidth: 80 }),
			],
			view: new View({
				projection: GRID,
				center: [25496580, 6673003],
				resolution: resolutionAt(16),
				maxResolution: resolutionAt(10),
				// The WMTS grid tops out at z17, so OL keeps serving those tiles
				// and magnifies them beyond that — no new detail, but z18–20 is
				// kinder on the eyes when judging a box edge.
				minResolution: resolutionAt(20),
			}),
		});
		wheelZoom = map
			.getInteractions()
			.getArray()
			.find((i) => i instanceof MouseWheelZoom) as MouseWheelZoom;

		// style: null — selected features keep the layer style; layerStyle
		// renders the highlight from `selected`, so click and N/P look identical.
		// layers: selection is scoped to the labels, so the AOI boundary (or any
		// future helper layer) can never be selected, resized, or deleted.
		select = new Select({ condition: click, style: null, layers: [labelLayer] });
		select.on('select', () => setSelected(select.getFeatures().getArray()[0] ?? null));
		map.addInteraction(select);

		// Clicking a neighbouring area opens it, so moving on does not mean
		// going back to the sidebar. A plain map listener rather than a second
		// Select: Select is scoped to the labels precisely so a boundary can
		// never be selected, dragged or deleted, and that should stay true.
		map.on('click', (event) => {
			// Three things outrank navigation. Drawing owns its clicks while it
			// is placing a box; a click that lands on a label is a selection;
			// and a click inside the area being labelled never leaves it,
			// whatever overlaps it — the collection has two overlapping pairs,
			// and losing an edit to a stray click on one would be indefensible.
			if (drawMode?.drawing) return;
			if (current && containsCoordinate(current.extent, event.coordinate)) return;
			let onLabel = false;
			let target: AoiInfo | undefined;
			map.forEachFeatureAtPixel(event.pixel, (feature, layer) => {
				if (layer === labelLayer) {
					onLabel = true;
					return true; // stop: the click belongs to the selection
				}
				if (layer === neighbourLayer && !target) {
					target = aois.find((a) => a.name === feature.get('name'));
				}
				return false;
			});
			if (!onLabel && target) void open(target);
		});
		showNeighbours();

		// Drag moves the selected box. Bound to the selection collection, so an
		// accidental drag on an unselected box pans the map instead of moving it.
		translate = new Translate({ features: select.getFeatures() });
		// Captured on start, while the box is still where it was: this is the
		// gesture most likely to go somewhere unintended.
		translate.on('translatestart', () => {
			if (selected) undoStack.edit(selected);
		});
		translate.on('translating', () => endHandles?.sync());
		translate.on('translateend', () => {
			if (selected) describe(selected);
			endHandles?.sync();
			markDirty();
		});
		map.addInteraction(translate);

		// Hand-drawing a new box. Owns the gesture, the sketch preview and its
		// listeners; the page answers onCommit with what a new box means.
		drawMode = new DrawMode({
			map,
			sketchSource,
			select,
			translate,
			wheelZoom,
			getWidth: () => width,
			onCommit: commitBox,
		});

		// Draggable ends of the selected box — the one-ended length edit. Owns
		// its handles, its raw pointer session and its teardown; the page only
		// answers its callbacks and pokes sync() after anything moves the box.
		endHandles = new EndHandles({
			map,
			layer: handleLayer,
			getSelected: () => selected,
			isDrawing: () => drawMode?.drawing ?? false,
			// Captured on start, while the box is still where it was, so one
			// undo reverts the whole drag.
			onGestureStart: (feature) => undoStack.edit(feature),
			onGestureMove: (feature) => {
				refreshMeasurements(feature);
				describe(feature);
			},
			onGestureEnd: () => markDirty(),
		});

		// Capture phase: this must run before OpenLayers' own wheel-zoom
		// listener, or resizing would also zoom the map. Plain scroll always
		// zooms — width moves behind Shift so reviewing with N/P can never
		// accidentally reshape a box.
		// Kept so it can be removed on destroy. onMount cannot do it: this
		// callback is async, and Svelte only honours a returned teardown from a
		// synchronous onMount — an async one always returns a Promise.
		onWheel = (event: WheelEvent) => {
			if (drawMode?.drawing) {
				event.preventDefault();
				event.stopPropagation();
				width = clampWidth(width + (event.deltaY < 0 ? WIDTH_STEP_M : -WIDTH_STEP_M));
				drawMode.preview();
			} else if (selected && event.shiftKey) {
				event.preventDefault();
				event.stopPropagation();
				resizeSelected(event.deltaY < 0 ? WIDTH_STEP_M : -WIDTH_STEP_M);
			}
		};
		map.getViewport().addEventListener('wheel', onWheel, {
			passive: false,
			capture: true,
		});

		const first = aois.find((a) => a.staged);
		if (first) await open(first);
	});

	onDestroy(() => {
		store.cancelPending();
		drawMode?.destroy();
		endHandles?.destroy();
		if (onWheel && map) {
			map.getViewport().removeEventListener('wheel', onWheel, { capture: true });
		}
	});
</script>

<svelte:window onkeydown={onKey} />

<div class="grid h-screen grid-cols-[22.5rem_1fr]">
	<aside class="flex min-h-0 flex-col border-r border-line bg-panel">
		<header class="border-b border-line px-3.5 pt-3.5 pb-3">
			<div class="flex items-baseline justify-between">
				<h1 class="text-sm font-semibold tracking-tight">
					rekka<span class="font-normal text-dim">·ai</span>
				</h1>
				<span class="font-mono text-[10px] tracking-[0.12em] text-dim uppercase">labelling</span>
			</div>

			<div class="mt-3 h-[3px] overflow-hidden rounded-full bg-line">
				<div
					class="h-full origin-left rounded-full bg-accent transition-transform duration-200 ease-[var(--ease-out)]"
					style="transform:scaleX({overall.total ? overall.reviewed / overall.total : 0})"
				></div>
			</div>
			<p class="mt-1.5 font-mono text-[11px] text-dim">
				<span class="text-fg">{overall.reviewed}</span>/{overall.total} reviewed
			</p>
		</header>

		<AreaTree {groups} {current} {loading} {loadError} onopen={open} />

		{#if current}
			<div class="border-t border-line px-3.5 py-3">
				{#if current.notes}
					<h3 class="text-xs font-semibold">{current.name}</h3>
					<p class="mt-1 text-xs leading-relaxed text-muted">{current.notes}</p>
				{/if}

				<ul class="mt-3 grid gap-[3px]">
					{#each [...CLASSES, 'rejected', 'added'] as key (key)}
						{@const n = store.counts[key as keyof typeof store.counts]}
						<li
							class="flex items-center gap-2 text-xs transition-opacity duration-150 {n
								? 'text-muted'
								: 'text-muted/40'}"
						>
							<i class="size-2 rounded-[2px]" style="background:{COLOURS[key]}"></i>
							{key}
							<b class="ml-auto font-mono text-[11px] font-medium text-fg">{n}</b>
						</li>
					{/each}
				</ul>
			</div>
		{/if}

		<KeyHelp bindings={BINDINGS} />
	</aside>

	<main class="relative flex min-w-0 flex-col">
		<div id="map" data-drawing={drawMode?.drawing ?? false} class="flex-1 bg-[#05070a]"></div>

		<ClassToolbar
			{activeClass}
			drawing={drawMode?.drawing ?? false}
			{width}
			hasSelection={selected !== null}
			saveState={store.saveState}
			saveError={store.saveError}
			onclassify={applyClass}
			ondelete={deleteSelected}
		/>

		{#if current && !loading && store.counts.total === 0}
			<div class="pointer-events-none absolute inset-0 flex items-center justify-center">
				<div
					class="rounded-lg border border-line bg-panel/90 px-5 py-4 text-center backdrop-blur-md"
				>
					<p class="text-sm text-fg">No candidates in {current.name} yet</p>
					<p class="mt-1 font-mono text-[11px] text-dim">bootstrap, then stage this area</p>
				</div>
			</div>
		{/if}

		<footer class="flex items-center gap-4 border-t border-line bg-panel px-3 py-1.5 text-xs">
			<span class="transition-colors duration-150 {drawMode?.drawing ? 'text-accent' : 'text-dim'}">
				{drawMode?.drawing
					? drawMode.phase === 0
						? 'drawing — click the nose'
						: drawMode.phase === 1
							? 'drawing — click the tail'
							: 'scroll to set width — click or Enter to place, Esc to redo'
					: selected
						? '⇧scroll: adjust width · drag: move · drag end: resize · Del: delete'
						: 'select a box, or press D to draw'}
			</span>
			<span class="ml-auto font-mono text-[11px] text-muted">{selectedInfo}</span>
		</footer>
	</main>
</div>

<style>
	#map[data-drawing='true'] {
		cursor: crosshair;
	}

	/* OpenLayers ships its own chrome; these pull it into the same palette. */
	:global(.ol-control) {
		background: none;
		padding: 0;
	}
	:global(.ol-zoom) {
		top: auto;
		bottom: 12px;
		left: 12px;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	:global(.ol-control button) {
		width: 26px;
		height: 26px;
		margin: 0;
		border: 1px solid var(--color-line);
		border-radius: 6px;
		background: color-mix(in srgb, var(--color-ink) 85%, transparent);
		backdrop-filter: blur(8px);
		color: var(--color-muted);
		font-size: 15px;
		line-height: 1;
		cursor: pointer;
		transition:
			background-color 140ms var(--ease-out),
			color 140ms var(--ease-out),
			transform 140ms var(--ease-out);
	}
	@media (hover: hover) and (pointer: fine) {
		:global(.ol-control button:hover) {
			background: var(--color-raised);
			color: var(--color-fg);
		}
	}
	:global(.ol-control button:active) {
		transform: scale(0.94);
	}
	:global(.ol-scale-line) {
		left: 46px;
		bottom: 12px;
		padding: 3px 7px 2px;
		border: 1px solid var(--color-line);
		border-radius: 6px;
		background: color-mix(in srgb, var(--color-ink) 85%, transparent);
		backdrop-filter: blur(8px);
	}
	:global(.ol-scale-line-inner) {
		border: 1px solid var(--color-muted);
		border-top: none;
		color: var(--color-fg);
		font-size: 10px;
		font-variant-numeric: tabular-nums;
		padding-bottom: 2px;
		will-change: auto;
	}
	:global(.ol-attribution) {
		right: 8px;
		bottom: 8px;
		font-size: 10px;
	}
	:global(.ol-attribution ul) {
		color: var(--color-dim);
		text-shadow: none;
	}
	:global(.ol-attribution button) {
		display: none;
	}

	@media (prefers-reduced-motion: reduce) {
		:global(.ol-control button:active) {
			transform: none;
		}
	}
</style>
