/**
 * Undo, pinned on what it is for: the edit that was not meant.
 *
 * Every change here autosaves within a second, so these tests stand in for the
 * moment of noticing — there is no uncommitted state to fall back on.
 */
import { describe, expect, test } from 'vitest';
import Feature from 'ol/Feature';
import Polygon from 'ol/geom/Polygon';
import VectorSource from 'ol/source/Vector';

import { UndoStack, COALESCE_MS } from './undo.svelte';
import { boxFromCentreline, type Coord } from './obb';

function box(properties: Record<string, unknown> = {}): Feature {
	const feature = new Feature(
		new Polygon([boxFromCentreline([25496000, 6673000], [25496016, 6673000], 3)]),
	);
	feature.setProperties({ status: 'candidate', class: '', length_m: 16, ...properties });
	return feature;
}

const ringOf = (f: Feature) => (f.getGeometry() as Polygon).getCoordinates()[0];

/** A clock the test drives, so the coalescing window needs no real waiting. */
function clock() {
	let t = 0;
	return { now: () => t, advance: (ms: number) => (t += ms) };
}

describe('undo', () => {
	test('puts a dragged box back exactly, not approximately', () => {
		// The reported worry: a box nudged somewhere it does not belong. An
		// inverse nudge would not do — rotation and scaling round, so replaying
		// one backwards lands millimetres off while looking correct.
		const source = new VectorSource();
		const feature = box();
		source.addFeature(feature);
		const before = JSON.stringify(ringOf(feature));

		const stack = new UndoStack();
		stack.edit(feature);
		(feature.getGeometry() as Polygon).setCoordinates([
			ringOf(feature).map(([x, y]) => [x + 4000, y - 2500]) as Coord[],
		]);
		expect(JSON.stringify(ringOf(feature))).not.toBe(before);

		expect(stack.undo(source)).toBe(feature);
		expect(JSON.stringify(ringOf(feature))).toBe(before);
	});

	test('restores the verdict a keystroke overwrote', () => {
		const source = new VectorSource();
		const feature = box({ status: 'candidate', class: '' });
		source.addFeature(feature);
		const stack = new UndoStack();

		stack.edit(feature);
		feature.set('status', 'rejected');
		stack.undo(source);

		expect(feature.get('status')).toBe('candidate');
		expect(feature.get('class')).toBe('');
	});

	test('a burst of arrow presses is one action, not sixty', () => {
		// Holding a key emits an edit per repeat. Undoing them one at a time is
		// not undo — the operator means "before I started nudging".
		const c = clock();
		const stack = new UndoStack(c.now);
		const source = new VectorSource();
		const feature = box();
		source.addFeature(feature);
		const before = JSON.stringify(ringOf(feature));

		for (let i = 0; i < 60; i++) {
			stack.edit(feature);
			c.advance(30); // key repeat, well inside the window
			(feature.getGeometry() as Polygon).setCoordinates([
				ringOf(feature).map(([x, y]) => [x + 0.1, y]) as Coord[],
			]);
		}
		expect(stack.depth).toBe(1);
		stack.undo(source);
		expect(JSON.stringify(ringOf(feature))).toBe(before);
	});

	test('a pause starts a new action, so the burst before it survives', () => {
		const c = clock();
		const stack = new UndoStack(c.now);
		stack.edit(box());
		c.advance(COALESCE_MS + 1);
		stack.edit(box());
		expect(stack.depth).toBe(2);
	});

	test('editing a different box is always its own action', () => {
		const c = clock();
		const stack = new UndoStack(c.now);
		stack.edit(box());
		c.advance(10);
		stack.edit(box());
		expect(stack.depth).toBe(2);
	});

	test('brings back a deleted box', () => {
		const source = new VectorSource();
		const feature = box({ status: 'confirmed', class: 'truck' });
		source.addFeature(feature);
		const stack = new UndoStack();

		stack.removed(feature, 0);
		source.removeFeature(feature);
		expect(source.getFeatures()).toHaveLength(0);

		expect(stack.undo(source)).toBe(feature);
		expect(source.getFeatures()).toEqual([feature]);
		expect(feature.get('class')).toBe('truck');
	});

	test('takes back a hand-drawn box, selecting nothing after', () => {
		const source = new VectorSource();
		const feature = box({ status: 'added', class: 'truck' });
		source.addFeature(feature);
		const stack = new UndoStack();

		stack.added(feature);
		expect(stack.undo(source)).toBeNull();
		expect(source.getFeatures()).toHaveLength(0);
	});

	test('undoing an empty stack is a no-op, not a crash', () => {
		expect(new UndoStack().undo(new VectorSource())).toBeNull();
	});

	test('is dropped when the area changes', () => {
		// Its entries point at features the source no longer holds; restoring
		// one would resurrect another area's box under this area's name.
		const stack = new UndoStack();
		stack.edit(box());
		stack.clear();
		expect(stack.depth).toBe(0);
		expect(stack.undo(new VectorSource())).toBeNull();
	});
});
