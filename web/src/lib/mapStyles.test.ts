/**
 * Map symbology, pinned where it carries meaning.
 *
 * Only the rules matter here, not the exact colours: a selected box must still
 * say what class it is, and the selection marker must never be mistakable for
 * a classification.
 */
import { describe, expect, test } from 'vitest';
import Feature from 'ol/Feature';
import Polygon from 'ol/geom/Polygon';
import type { Stroke } from 'ol/style';

import { COLOURS, selectedStyleFor, styleFor } from './mapStyles';
import { boxFromCentreline } from './obb';

function box(properties: Record<string, unknown>): Feature {
	const feature = new Feature(
		new Polygon([boxFromCentreline([25496000, 6673000], [25496016, 6673000], 3)]),
	);
	feature.setProperties({ length_m: 16, width_m: 3, ...properties });
	return feature;
}

const strokeColours = (feature: Feature): string[] =>
	selectedStyleFor(feature)
		.map((s) => (s.getStroke() as Stroke | null)?.getColor())
		.filter((c): c is string => typeof c === 'string');

describe('selection', () => {
	test('keeps the class colour visible', () => {
		// The bug this guards: selection used to replace the stroke, so every
		// selected box looked identical whatever verdict it carried.
		for (const klass of ['truck', 'bus', 'van'] as const) {
			const colours = strokeColours(box({ status: 'confirmed', class: klass }));
			expect(colours, klass).toContain(COLOURS[klass]);
		}
	});

	test('marks selection with a colour no class uses', () => {
		// Otherwise the marker reads as a classification.
		const classColours = Object.values(COLOURS);
		const selected = strokeColours(box({ status: 'confirmed', class: 'truck' }));
		const marker = selected.filter((c) => !classColours.includes(c));
		expect(marker.length).toBeGreaterThan(0);
	});

	test('adds no second fill over the one styleFor already draws', () => {
		// Two translucent fills stacked is what made the highlight muddy.
		const fills = selectedStyleFor(box({ status: 'candidate', class: '' }))
			.map((s) => s.getFill())
			.filter(Boolean);
		expect(fills).toHaveLength(1);
	});

	test('labels the long and short edges with their measurements', () => {
		const texts = selectedStyleFor(box({ status: 'confirmed', class: 'truck' }))
			.map((s) => s.getText()?.getText())
			.filter(Boolean);
		expect(texts).toEqual(['16 m', '3 m']);
	});
});

describe('unselected boxes', () => {
	test('an unreviewed candidate is not drawn in any class colour', () => {
		const colour = styleFor(box({ status: 'candidate', class: '' }))
			.getStroke()
			?.getColor();
		expect(colour).toBe(COLOURS.candidate);
	});

	test('a reject is dashed, so it reads as struck through', () => {
		const stroke = styleFor(box({ status: 'rejected', class: 'truck' })).getStroke();
		expect(stroke?.getLineDash()).toBeTruthy();
		expect(stroke?.getColor()).toBe(COLOURS.rejected);
	});
});
