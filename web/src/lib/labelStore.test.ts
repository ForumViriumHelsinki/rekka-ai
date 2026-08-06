/**
 * The save path, which is the one place this tool can lose real work.
 *
 * These tests exist because the logic was extracted out of a 1000-line
 * component where it could not be exercised at all. Each one pins an invariant
 * the labelling workflow depends on rather than an implementation detail.
 */
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import Feature from 'ol/Feature';
import Polygon from 'ol/geom/Polygon';

import { GRID_CRS } from './grid';
import { LabelStore, RETRY_DELAY_MS, SAVE_DEBOUNCE_MS } from './labelStore.svelte';
import { boxFromCentreline } from './obb';

function box(properties: Record<string, unknown> = {}): Feature {
	const feature = new Feature(
		new Polygon([boxFromCentreline([25496000, 6673000], [25496016, 6673000], 3)]),
	);
	feature.setProperties({ status: 'candidate', class: '', ...properties });
	return feature;
}

let store: LabelStore;
let ok: () => Response;

beforeEach(() => {
	vi.useFakeTimers();
	ok = () => ({ ok: true, status: 200, json: async () => ({}) }) as Response;
	store = new LabelStore();
	store.area = 'yard';
});

afterEach(() => {
	store.cancelPending();
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

describe('recount', () => {
	test('tallies verdicts and classes', () => {
		store.source.addFeatures([
			box(),
			box({ status: 'confirmed', class: 'truck' }),
			box({ status: 'added', class: 'van' }),
			box({ status: 'rejected', class: '' }),
		]);
		store.recount();
		expect(store.counts).toMatchObject({
			total: 4,
			reviewed: 3,
			truck: 1,
			van: 1,
			added: 1,
			rejected: 1,
		});
	});

	test('a rejected box does not count towards its old class', () => {
		// Rejects keep whatever class they were given before the X; counting
		// them would inflate the class tallies the operator works against.
		store.source.addFeature(box({ status: 'rejected', class: 'truck' }));
		store.recount();
		expect(store.counts.truck).toBe(0);
		expect(store.counts.rejected).toBe(1);
	});

	test('reports counts to the subscriber', () => {
		const seen: number[] = [];
		const watched = new LabelStore({ onCounts: (c) => seen.push(c.total) });
		watched.source.addFeature(box());
		watched.recount();
		expect(seen).toEqual([1]);
	});
});

describe('save', () => {
	test('clears dirty when the write succeeds', async () => {
		vi.stubGlobal('fetch', vi.fn(ok));
		store.source.addFeature(box());
		store.markDirty();
		expect(store.dirty).toBe(true);
		await store.save();
		expect(store.dirty).toBe(false);
		expect(store.saveError).toBe('');
	});

	test('stays dirty and retries when the write fails', async () => {
		const fetchMock = vi.fn(async () => ({ ok: false, status: 500 }) as Response);
		vi.stubGlobal('fetch', fetchMock);
		store.source.addFeature(box());
		store.markDirty();
		await store.save();

		// The failure must be loud, and the work must still count as unsaved:
		// a tool that forgets a failed write is worse than one that crashes.
		expect(store.dirty).toBe(true);
		expect(store.saveError).toBe('HTTP 500');
		expect(store.saveState).toBe('SAVE FAILED');

		fetchMock.mockImplementation(async () => ok());
		await vi.advanceTimersByTimeAsync(RETRY_DELAY_MS);
		expect(fetchMock).toHaveBeenCalledTimes(2);
		expect(store.saveError).toBe('');
		expect(store.dirty).toBe(false);
	});

	test('does nothing without an area', async () => {
		const fetchMock = vi.fn(ok) as unknown as ReturnType<typeof vi.fn>;
		vi.stubGlobal('fetch', fetchMock);
		store.area = null;
		await store.save();
		expect(fetchMock).not.toHaveBeenCalled();
	});

	test('recomputes measurements from geometry before writing', async () => {
		const fetchMock = vi.fn(ok) as unknown as ReturnType<typeof vi.fn>;
		vi.stubGlobal('fetch', fetchMock);
		// A stale length, as an editor that moved a vertex would leave behind.
		const feature = box({ length_m: 999, width_m: 999 });
		store.source.addFeature(feature);
		await store.save();
		expect(feature.get('length_m')).toBeCloseTo(16, 2);
		expect(feature.get('width_m')).toBeCloseTo(3, 2);
	});

	test('writes EPSG:3879 metres, and says so', async () => {
		const fetchMock = vi.fn(ok) as unknown as ReturnType<typeof vi.fn>;
		vi.stubGlobal('fetch', fetchMock);
		store.source.addFeature(box({ status: 'added', class: 'truck' }));
		await store.save();
		const body = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
		// A reader that trusts RFC 7946 would put these boxes off Africa, so
		// the `crs` member is not decoration — it is the whole declaration.
		expect(body.crs).toEqual(GRID_CRS);
		const [easting, northing] = body.features[0].geometry.coordinates[0][0];
		// Helsinki in grid metres — not the ~25° a WGS84 write would give.
		expect(easting).toBeCloseTo(25496000, -2);
		expect(northing).toBeCloseTo(6673000, -2);
	});
});

describe('the pending save is cancelled before every other write', () => {
	test('save() cancels the debounce, so it cannot fire twice', async () => {
		const fetchMock = vi.fn(ok) as unknown as ReturnType<typeof vi.fn>;
		vi.stubGlobal('fetch', fetchMock);
		store.source.addFeature(box());
		store.markDirty(); // schedules a save
		await store.save(); // the operator hit Ctrl+S first
		await vi.advanceTimersByTimeAsync(SAVE_DEBOUNCE_MS * 2);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	test('load() cancels the debounce, so it cannot write under the new name', async () => {
		// The failure this prevents: edit `yard`, switch to `rigs` before the
		// debounce fires, and the queued timer PUTs yard's features to rigs.
		const fetchMock = vi.fn(async (url: string, init?: { method?: string }) => {
			void init;
			if (String(url).includes('rigs')) {
				return {
					ok: true,
					status: 200,
					json: async () => ({ type: 'FeatureCollection', features: [] }),
				} as Response;
			}
			return ok();
		});
		vi.stubGlobal('fetch', fetchMock);

		store.source.addFeature(box());
		store.markDirty();
		await store.load('rigs');
		await vi.advanceTimersByTimeAsync(SAVE_DEBOUNCE_MS * 2);

		const puts = fetchMock.mock.calls.filter((c) => c[1]?.method === 'PUT');
		expect(puts).toHaveLength(0);
	});
});

describe('load', () => {
	test('replaces the features and resets the dirty flag', async () => {
		store.source.addFeature(box());
		store.dirty = true;
		vi.stubGlobal(
			'fetch',
			vi.fn(async () => ({
				ok: true,
				status: 200,
				json: async () => ({
					type: 'FeatureCollection',
					crs: GRID_CRS,
					features: [
						{
							type: 'Feature',
							geometry: {
								type: 'Polygon',
								coordinates: [
									[
										[25496000, 6673000],
										[25496016, 6673000],
										[25496016, 6673002.5],
										[25496000, 6673002.5],
										[25496000, 6673000],
									],
								],
							},
							properties: { status: 'confirmed', class: 'truck' },
						},
					],
				}),
			})),
		);
		await store.load('rigs');
		expect(store.area).toBe('rigs');
		expect(store.source.getFeatures()).toHaveLength(1);
		expect(store.counts.truck).toBe(1);
		expect(store.dirty).toBe(false);
	});

	test('a failed load raises rather than silently emptying the area', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn(async () => ({ ok: false, status: 404 }) as Response),
		);
		store.source.addFeature(box());
		await expect(store.load('nope')).rejects.toThrow('HTTP 404');
		// The features that were open are still there to be saved.
		expect(store.source.getFeatures()).toHaveLength(1);
	});
});

describe('round trip', () => {
	// Two features exactly as `labels.write` leaves them: EPSG:3879 metres at
	// COORD_DECIMALS, including an integral coordinate (which JSON.stringify
	// and json.dumps only agree on because Python normalizes 175.0 to 175).
	// Two, because one cannot catch a reordering.
	const FIRST_RING = [
		[25496000.123, 6673000.877],
		[25496016.123, 6673000.877],
		[25496016.123, 6673003.377],
		[25496000.123, 6673003.377],
		[25496000.123, 6673000.877],
	];
	const SECOND_RING = [
		[25496100, 6673100.5],
		[25496112.25, 6673100.5],
		[25496112.25, 6673103],
		[25496100, 6673103],
		[25496100, 6673100.5],
	];
	const FIXTURE = {
		type: 'FeatureCollection',
		crs: GRID_CRS,
		features: [
			{
				type: 'Feature',
				geometry: { type: 'Polygon', coordinates: [FIRST_RING] },
				properties: { aoi: 'kamppi', status: 'candidate', class: '', length_m: 12.34 },
			},
			{
				type: 'Feature',
				geometry: { type: 'Polygon', coordinates: [SECOND_RING] },
				properties: { aoi: 'kamppi', status: 'candidate', class: '', length_m: 56.78 },
			},
		],
	};

	function stubLoadAndCapturePut(): ReturnType<typeof vi.fn> {
		const fetchMock = vi.fn(async (url: string, init?: { method?: string }) => {
			if (init?.method === 'PUT') return ok();
			return { ok: true, status: 200, json: async () => FIXTURE } as Response;
		});
		vi.stubGlobal('fetch', fetchMock);
		return fetchMock;
	}

	function putBody(fetchMock: ReturnType<typeof vi.fn>) {
		const put = fetchMock.mock.calls.find((c) => c[1]?.method === 'PUT');
		return JSON.parse(put![1]!.body as string);
	}

	test('loading and saving without an edit reproduces the file exactly', async () => {
		// The regression this whole CRS choice exists for: an operator opens an
		// area, changes nothing, and the file must come back identical — not
		// merely equivalent to eight decimal places.
		const fetchMock = stubLoadAndCapturePut();
		await store.load('kamppi');
		await store.save();
		expect(JSON.stringify(putBody(fetchMock))).toBe(JSON.stringify(FIXTURE));
	});

	test('a verdict edit changes the properties it touched and nothing else', async () => {
		const fetchMock = stubLoadAndCapturePut();
		await store.load('kamppi');
		const feature = store.source.getFeatures().find((f) => f.get('length_m') === 12.34)!;
		feature.set('status', 'confirmed');
		feature.set('class', 'truck');
		await store.save();
		const body = putBody(fetchMock);
		expect(body.features[0].geometry.coordinates).toEqual([FIRST_RING]);
		expect(body.features[0].properties.status).toBe('confirmed');
		expect(body.features[0].properties.class).toBe('truck');
		// Untouched geometry is not re-measured, so a JS/Python disagreement in
		// the last rounded digit cannot creep into a file nobody edited.
		expect(body.features[0].properties.length_m).toBe(12.34);
		expect(body.features[1]).toEqual(FIXTURE.features[1]);
	});

	test('a geometry edit recomputes measurements, without internal keys', async () => {
		const fetchMock = stubLoadAndCapturePut();
		await store.load('kamppi');
		const feature = store.source.getFeatures().find((f) => f.get('length_m') === 12.34)!;
		const geometry = feature.getGeometry() as Polygon;
		// Stretch the 16 m box to 20 m along its long axis.
		const ring = (geometry.getCoordinates()[0] as number[][]).map(([x, y], i) => [
			i === 1 || i === 2 ? x + 4 : x,
			y,
		]);
		geometry.setCoordinates([ring]);
		await store.save();
		const written = putBody(fetchMock).features[0];
		expect(written.properties.length_m).toBeCloseTo(20, 2);
		expect(Object.keys(written.properties)).not.toContain('rekka:geomDirty');
		expect(Object.keys(written.properties)).not.toContain('rekka:order');
	});

	test('a hand-drawn box is measured and appended after the file features', async () => {
		const fetchMock = stubLoadAndCapturePut();
		await store.load('kamppi');
		store.source.addFeature(box({ status: 'added', class: 'truck' }));
		await store.save();
		const body = putBody(fetchMock);
		expect(body.features).toHaveLength(3);
		expect(body.features[2].properties.class).toBe('truck');
		// Never measured before, so it must be measured now.
		expect(body.features[2].properties.length_m).toBeCloseTo(16, 2);
	});
});
