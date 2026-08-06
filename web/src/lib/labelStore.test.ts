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

	test('writes WGS84, as RFC 7946 requires', async () => {
		const fetchMock = vi.fn(ok) as unknown as ReturnType<typeof vi.fn>;
		vi.stubGlobal('fetch', fetchMock);
		store.source.addFeature(box({ status: 'added', class: 'truck' }));
		await store.save();
		const body = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
		const [lon, lat] = body.features[0].geometry.coordinates[0][0];
		// Helsinki, in degrees — not the 2.5e7 easting of EPSG:3879.
		expect(lon).toBeGreaterThan(24);
		expect(lon).toBeLessThan(26);
		expect(lat).toBeGreaterThan(59);
		expect(lat).toBeLessThan(61);
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
					features: [
						{
							type: 'Feature',
							geometry: {
								type: 'Polygon',
								coordinates: [
									[
										[24.93, 60.17],
										[24.9301, 60.17],
										[24.9301, 60.1701],
										[24.93, 60.1701],
										[24.93, 60.17],
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
