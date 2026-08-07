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
import Point from 'ol/geom/Point';
import type { Stroke, Style } from 'ol/style';

import { COLOURS, aoiStyle, neighbourStyle, selectedStyleFor, styleFor } from './mapStyles';
import {
	boxFromCentreline,
	centrelineOf,
	rotateCentreline,
	scaleCentreline,
	ROTATE_STEP_DEG,
	LENGTH_STEP_M,
	type Coord,
} from './obb';

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

	test('marks selection with a wider class-coloured stroke, not a new colour', () => {
		// The amber halo is gone: amber is the candidate class colour, so a
		// selected candidate read as amber-on-amber. Selection is carried by
		// the width bump — the handles and edge labels only exist on a
		// selection, and they do the rest of the marking.
		for (const klass of ['truck', 'bus', 'van'] as const) {
			const feature = box({ status: 'confirmed', class: klass });
			const bumped = selectedStyleFor(feature)
				.filter((s) => s.getStroke()?.getColor() === COLOURS[klass])
				.map((s) => s.getStroke()!.getWidth()!);
			expect(bumped, klass).toHaveLength(1);
			expect(bumped[0], klass).toBeGreaterThan(styleFor(feature).getStroke()!.getWidth()!);
		}
	});

	test('keeps the dark casing, because one thin stroke is lost on an orthophoto', () => {
		const [casing] = selectedStyleFor(box({ status: 'confirmed', class: 'truck' }));
		expect(casing.getStroke()?.getColor()).toContain('8,10,14');
		expect(casing.getStroke()!.getWidth()!).toBeGreaterThan(4);
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

describe('a neighbouring area', () => {
	const neighbour = () => {
		const feature = new Feature(
			new Polygon([
				[
					[0, 0],
					[10, 0],
					[10, 10],
					[0, 10],
					[0, 0],
				],
			]),
		);
		feature.set('name', 'kylasaari');
		return neighbourStyle(feature);
	};

	const alpha = (colour: string) => Number(colour.match(/,\s*([\d.]+)\s*\)$/)?.[1] ?? 1);
	/** The bright line, not the dark rail under it: the widest-drawn stroke is
	 * the casing, and the one after it is what the eye actually reads. */
	const accent = (styles: Style[]) => styles[styles.length - 1].getStroke()!;

	test('is named, so clicking it is navigation and not a guess', () => {
		const texts = neighbour()
			.map((s) => s.getText()?.getText())
			.filter(Boolean);
		expect(texts).toEqual(['kylasaari']);
	});

	test('is cased, because a single thin line is lost against an orthophoto', () => {
		// The regression this guards: drawn as one 1 px line it was invisible
		// over bright concrete, which is most of a yard. Dimming it further is
		// the wrong lever — contrast comes from the dark rail underneath.
		const casing = neighbour()[0].getStroke()!;
		expect(casing.getColor()).toContain('0,0,0');
		expect(casing.getWidth()!).toBeGreaterThan(accent(neighbour()).getWidth()!);
	});

	test('is quieter than the boundary of the area being labelled', () => {
		// "Where I am" versus "where I could go" has to survive a glance, and
		// both are cased — so the hierarchy has to live in the bright line.
		const here = accent(aoiStyle);
		const there = accent(neighbour());
		expect(there.getWidth()!).toBeLessThan(here.getWidth()!);
		expect(alpha(there.getColor() as string)).toBeLessThan(alpha(here.getColor() as string));
	});

	test('carries a fill, or the whole rectangle is not a click target', () => {
		// Hit-testing a stroke-only polygon only hits the outline.
		expect(neighbour().some((s) => s.getFill())).toBe(true);
	});

	test('is drawn in no class colour, so it cannot read as a verdict', () => {
		const classColours = Object.values(COLOURS);
		for (const style of neighbour()) {
			expect(classColours).not.toContain(style.getStroke()?.getColor());
			expect(classColours).not.toContain(style.getFill()?.getColor());
		}
	});
});

describe('measurement labels while a box is nudged', () => {
	/** The bug: the labels used to rotate with their edge, folded upright past
	 * 90 deg. The fold is a discontinuity, and it sat at headings 0 and 90 --
	 * where vehicles actually park -- so an arrow-key nudge across the line
	 * mirrored a label through 180 deg, every press, back and forth. */
	test('never flip as the box turns through a full revolution', () => {
		let nose: Coord = [25496000, 6673000];
		let tail: Coord = [25496016, 6673000];
		const seen = new Set<number | undefined>();
		for (let i = 0; i < 900; i++) {
			const feature = new Feature(new Polygon([boxFromCentreline(nose, tail, 3)]));
			feature.setProperties({ length_m: 16, width_m: 3, status: 'confirmed', class: 'truck' });
			for (const style of selectedStyleFor(feature)) {
				if (style.getText()) seen.add(style.getText()!.getRotation());
			}
			[nose, tail] = rotateCentreline(nose, tail, 0.2);
		}
		// One value across every heading: no angle, so nothing to jump.
		expect([...seen]).toEqual([undefined]);
	});

	test('keep their side across repeated nudges', () => {
		// The bug, exactly as reported: every arrow press threw the labels from
		// the top of the box to the bottom and back — on rotation and on length
		// alike — because a rebuilt ring lists the same corners in a different
		// order, and the labels followed the ring rather than the ground.
		const sideOf = (ring: Coord[]) => {
			const feature = new Feature(new Polygon([ring]));
			feature.setProperties({ length_m: 16, width_m: 3, status: 'confirmed', class: 'truck' });
			const [first] = selectedStyleFor(feature)
				.filter((s) => s.getText()?.getText() === '16 m')
				.map((s) => (s.getGeometry() as Point).getCoordinates());
			return first;
		};
		for (const [what, nudge] of [
			['rotation', (n: Coord, t: Coord) => rotateCentreline(n, t, ROTATE_STEP_DEG)],
			['length', (n: Coord, t: Coord) => scaleCentreline(n, t, LENGTH_STEP_M, 3)],
		] as const) {
			let ring = boxFromCentreline([25496000, 6673000], [25496016, 6673000], 3) as Coord[];
			const sides: number[] = [];
			for (let i = 0; i < 8; i++) {
				sides.push(sideOf(ring)[1]);
				ring = boxFromCentreline(...nudge(...centrelineOf(ring)), 3) as Coord[];
			}
			// Swapping sides moves the label by a whole box width. Rotation drags
			// it a few centimetres, so anything near 3 m is a flip and nothing else.
			const spread = Math.max(...sides) - Math.min(...sides);
			expect(spread, `${what}: ${sides.join(' ')}`).toBeLessThan(1);
		}
	});

	test('sit on the edge they measure, which is what names them', () => {
		const feature = box({ status: 'confirmed', class: 'truck' });
		const ring = (feature.getGeometry() as Polygon).getCoordinates()[0] as Coord[];
		const mids = selectedStyleFor(feature)
			.filter((s) => s.getText())
			.map((s) => (s.getGeometry() as Point).getCoordinates());
		expect(mids).toHaveLength(2);
		for (const [mx, my] of mids) {
			const onAnEdge = [0, 1, 2, 3].some((i) => {
				const ex = (ring[i][0] + ring[i + 1][0]) / 2;
				const ey = (ring[i][1] + ring[i + 1][1]) / 2;
				return Math.hypot(mx - ex, my - ey) < 1e-6;
			});
			expect(onAnEdge).toBe(true);
		}
	});
});
