/**
 * One area's labels, and getting them back to disk.
 *
 * Split out of the page because this is where the project's data can actually
 * be lost. Everything else on the page redraws if it goes wrong; a correction
 * that never reaches `labels/` is human hours gone, and the file is the one
 * artifact the pipeline cannot regenerate. Isolated, it is testable — which is
 * the point.
 *
 * Three rules it exists to keep:
 *
 * - **A failed save stays dirty.** It reports the failure and retries rather
 *   than quietly dropping the work. The usual cause is a restarted dev server.
 * - **A pending save is cancelled before any other write.** Otherwise a queued
 *   debounce fires after the operator has switched areas and writes this area's
 *   features under the next area's name.
 * - **Measurements are recomputed from geometry on the way out**, never
 *   trusted from memory: a dragged vertex leaves a stale `length_m` otherwise.
 *   But only where the geometry actually changed — recomputing `length_m` for
 *   an untouched box risks this and Python disagreeing in the last rounded
 *   digit, which is a diff nobody asked for.
 * - **The file is written in the map's own CRS.** Label files store EPSG:3879
 *   (see `geo.crs_member` on the Python side), so reading and writing move no
 *   coordinates at all: OpenLayers skips the transform when the data and
 *   feature projections match, and a `COORD_DECIMALS` write reproduces the
 *   digits already on disk. Round-tripping through WGS84 instead re-derived
 *   every coordinate in the file on every save, so changing one box's class
 *   rewrote all of them.
 */
import GeoJSON from 'ol/format/GeoJSON';
import type Polygon from 'ol/geom/Polygon';
import VectorSource from 'ol/source/Vector';
import Feature from 'ol/Feature';

import { COORD_DECIMALS, GRID, GRID_CRS } from '$lib/grid';
import { measure, type Coord } from '$lib/obb';

/** Internal feature properties, never written to disk. */
const GEOM_DIRTY = 'rekka:geomDirty';
/** A feature's zero-based position in the file it was loaded from. Stamped at
 * load and stripped before write. Exported because it is the only stable way
 * to address a box: `VectorSource.getFeatures()` iterates a spatial index, so
 * its array order is neither the file's nor stable across edits — anything
 * that shows or takes a box number must go through this, not through an
 * index into that array. Undefined on a hand-drawn box until it is reloaded. */
export const ORDER = 'rekka:order';

/** Long enough to absorb a burst of keystrokes, short enough to feel saved. */
export const SAVE_DEBOUNCE_MS = 800;
/** A failed save is usually a restarted dev server; keep trying. */
export const RETRY_DELAY_MS = 5000;

export type Counts = {
	total: number;
	reviewed: number;
	truck: number;
	bus: number;
	van: number;
	car: number;
	rejected: number;
	added: number;
};

export function emptyCounts(): Counts {
	return { total: 0, reviewed: 0, truck: 0, bus: 0, van: 0, car: 0, rejected: 0, added: 0 };
}

/** Recompute a box's measurements from its geometry.
 *
 * Exported because the page edits geometry directly (drag, wheel, arrows) and
 * must refresh as it goes; the store also does it on save for every feature
 * whose geometry changed, so nothing stale can reach disk even if a caller
 * forgets. */
export function refreshMeasurements(feature: Feature): void {
	const ring = (feature.getGeometry() as Polygon).getCoordinates()[0] as Coord[];
	const m = measure(ring);
	feature.set('length_m', Math.round(m.length * 100) / 100, true);
	feature.set('width_m', Math.round(m.width * 100) / 100, true);
	feature.set('heading_deg', Math.round(m.heading * 10) / 10, true);
}

export class LabelStore {
	/** The features being edited. The map renders straight from this. */
	readonly source = new VectorSource();

	area = $state<string | null>(null);
	dirty = $state(false);
	saving = $state(false);
	saveError = $state('');
	counts = $state<Counts>(emptyCounts());

	#timer: ReturnType<typeof setTimeout> | null = null;
	#format = new GeoJSON();
	#onCounts?: (counts: Counts) => void;

	/** `onCounts` fires after every recount, so a sidebar can follow along
	 * without this module knowing one exists. */
	constructor(options: { onCounts?: (counts: Counts) => void } = {}) {
		this.#onCounts = options.onCounts;
	}

	get saveState(): 'saving' | 'SAVE FAILED' | 'unsaved' | 'saved' {
		if (this.saving) return 'saving';
		if (this.saveError) return 'SAVE FAILED';
		return this.dirty ? 'unsaved' : 'saved';
	}

	/** Drop any queued save. Called before every write, and on teardown. */
	cancelPending(): void {
		if (this.#timer) {
			clearTimeout(this.#timer);
			this.#timer = null;
		}
	}

	/** Replace the contents with one area's labels, fetched from the API. */
	async load(name: string): Promise<void> {
		this.cancelPending();
		const response = await fetch(`/api/labels/${name}`);
		if (!response.ok) throw new Error(`HTTP ${response.status}`);
		const collection = await response.json();
		this.area = name;
		this.source.clear();
		if (collection.features?.length) {
			const features = this.#format.readFeatures(collection, {
				featureProjection: GRID,
				dataProjection: GRID,
			});
			// Remember each feature's position in the file, and watch its
			// geometry. VectorSource holds features in a spatial index whose
			// iteration order is arbitrary, so saving in that order would
			// reshuffle the whole file; and only a geometry that actually moved
			// needs its measurements recomputed.
			features.forEach((feature, i) => {
				feature.set(ORDER, i, true);
				feature.getGeometry()?.on('change', () => feature.set(GEOM_DIRTY, true, true));
			});
			this.source.addFeatures(features);
		}
		this.recount();
		this.dirty = false;
		this.saveError = '';
	}

	/** Something changed: recount, and schedule the debounced save. */
	markDirty(): void {
		this.dirty = true;
		this.recount();
		this.cancelPending();
		this.#timer = setTimeout(() => void this.save(), SAVE_DEBOUNCE_MS);
	}

	async save(): Promise<void> {
		if (!this.area) return;
		// Before anything else: a queued debounce must not fire after the
		// operator has moved on and write these features under another name.
		this.cancelPending();
		this.saving = true;
		this.saveError = '';
		// File order first (VectorSource's spatial index iterates arbitrarily),
		// hand-drawn additions last.
		const features = this.source
			.getFeatures()
			.slice()
			.sort(
				(a, b) =>
					((a.get(ORDER) as number | undefined) ?? Infinity) -
					((b.get(ORDER) as number | undefined) ?? Infinity),
			);
		const out = features.map((feature) => {
			// Edited geometry, or a hand-drawn box that has never been measured.
			// `=== undefined`, not falsy: the first feature in the file is 0.
			if (feature.get(GEOM_DIRTY) || feature.get(ORDER) === undefined) {
				refreshMeasurements(feature);
			}
			// Serialize through a clone so the internal bookkeeping keys never
			// reach disk.
			const clone = new Feature(feature.getGeometry());
			for (const [key, value] of Object.entries(feature.getProperties())) {
				if (key === GEOM_DIRTY || key === ORDER || key === 'geometry') continue;
				clone.set(key, value, true);
			}
			return this.#format.writeFeatureObject(clone, {
				featureProjection: GRID,
				dataProjection: GRID,
				decimals: COORD_DECIMALS,
			});
		});
		// `crs` first, as `labels.write` orders it: the two writers take turns
		// on these files and a key-order difference is a diff of its own.
		const geojson = { type: 'FeatureCollection', crs: GRID_CRS, features: out };
		try {
			const response = await fetch(`/api/labels/${this.area}`, {
				method: 'PUT',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify(geojson),
			});
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			this.dirty = false;
		} catch (e) {
			this.saveError = e instanceof Error ? e.message : String(e);
			this.#timer = setTimeout(() => void this.save(), RETRY_DELAY_MS);
		} finally {
			this.saving = false;
		}
	}

	recount(): void {
		const features = this.source.getFeatures();
		const next = emptyCounts();
		next.total = features.length;
		for (const feature of features) {
			const status = feature.get('status') ?? 'candidate';
			if (status !== 'candidate') next.reviewed += 1;
			if (status === 'rejected') next.rejected += 1;
			// Hand-drawn boxes the detector missed — the number that measures
			// zero-shot recall, so it is worth watching on its own.
			if (status === 'added') next.added += 1;
			const klass = feature.get('class');
			if (status !== 'rejected' && klass && klass in next) {
				(next as unknown as Record<string, number>)[klass] += 1;
			}
		}
		this.counts = next;
		this.#onCounts?.(next);
	}
}
