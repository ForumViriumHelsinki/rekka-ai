/**
 * How labels look on the map.
 *
 * Pure functions of a feature to an OpenLayers style — no component state, so
 * they can be reasoned about (and tested) without a map. The palette lives
 * here too: these five colours are map *symbology*, and deliberately share
 * nothing with the interface chrome, so a colour on the map always means a
 * classification and never a selection.
 */
import { Fill, Stroke, Style, Text } from 'ol/style';
import Point from 'ol/geom/Point';
import Polygon from 'ol/geom/Polygon';
import type Feature from 'ol/Feature';
import type { FeatureLike } from 'ol/Feature';

import type { Coord } from '$lib/obb';

export const COLOURS: Record<string, string> = {
	candidate: '#f0a726',
	truck: '#3ddc84',
	bus: '#4aa3ff',
	van: '#c07cf5',
	rejected: '#ff5a5a',
	// Hand-drawn boxes render in their class colour; the swatch is symbolic.
	added: '#94a3b8',
};

export const sketchStyle = new Style({
	stroke: new Stroke({ color: '#ffc14d', width: 2, lineDash: [6, 6] }),
	fill: new Fill({ color: 'rgba(255,180,60,0.08)' }),
});

// A casing, not a single line: one stroke can always be lost against some
// part of an orthophoto — pale on concrete, dark in shadow — and a yard is
// mostly bright concrete. The dark rail carries the bright dashes over both.
// Neutral by design: amber is chrome and the five class colours are
// symbology, so the boundary must borrow from neither.
export const aoiStyle = [
	new Style({ stroke: new Stroke({ color: 'rgba(0,0,0,0.5)', width: 4.5 }) }),
	new Style({
		stroke: new Stroke({ color: 'rgba(255,255,255,0.95)', width: 2, lineDash: [10, 8] }),
	}),
];

/** Another area's boundary: findable enough to aim at, quiet enough to ignore.
 *
 * Cased like `aoiStyle`, and for the same reason — a single thin line is what
 * an orthophoto eats. Being *dimmer* than the current boundary does not make
 * it readable at 1 px; it just makes it invisible over bright concrete, which
 * is most of a yard. So the hierarchy is carried by the bright line on top
 * (narrower, less opaque, tighter dashes) while the dark rail underneath keeps
 * both boundaries legible over anything.
 *
 * Neutral, like the current boundary: a boundary is neither chrome nor
 * classification, so it borrows from neither palette.
 *
 * The fill is almost nothing, but not nothing: it is what makes the whole
 * rectangle a click target instead of an outline to hit. */
export function neighbourStyle(feature: FeatureLike): Style[] {
	return [
		new Style({
			stroke: new Stroke({ color: 'rgba(0,0,0,0.5)', width: 4 }),
			fill: new Fill({ color: 'rgba(255,255,255,0.03)' }),
		}),
		new Style({
			stroke: new Stroke({ color: 'rgba(255,255,255,0.7)', width: 1.75, lineDash: [6, 7] }),
			// Named, because clicking an unlabelled rectangle to be taken
			// somewhere unknown is not navigation.
			text: new Text({
				text: String(feature.get('name') ?? ''),
				font: '600 11px ui-monospace, monospace',
				fill: new Fill({ color: 'rgba(255,255,255,0.75)' }),
				backgroundFill: new Fill({ color: 'rgba(5,7,10,0.6)' }),
				padding: [1, 4, 1, 4],
			}),
		}),
	];
}

export function styleFor(feature: Feature): Style {
	const status = feature.get('status') ?? 'candidate';
	const klass = feature.get('class') ?? '';
	const rejected = status === 'rejected';
	const colour = rejected ? COLOURS.rejected : klass ? COLOURS[klass] : COLOURS.candidate;
	return new Style({
		stroke: new Stroke({
			color: colour,
			width: rejected ? 1.5 : 2.5,
			lineDash: rejected ? [5, 5] : undefined,
		}),
		fill: new Fill({
			color: status === 'candidate' ? 'rgba(240,167,38,0.10)' : 'rgba(0,0,0,0.01)',
		}),
	});
}

/** Selection must be unmistakable at a glance: N/P jumps between candidates
 * dozens of times a session, and "which box am I on" cannot be a guess.
 *
 * A halo *under* the box rather than a stroke replacing it, so the class
 * colour still reads while the box is selected. Covering it up meant every
 * selected box looked alike whatever its verdict, which is why labelling
 * used to drop the highlight to reveal the colour — and that in turn left
 * the box uneditable until it was clicked again. The halo is amber, never
 * a class colour, so it still cannot be read as a classification.
 *
 * Rendered by the layer itself (see the page's layerStyle), not the Select
 * interaction's style: the Select overlay is unreliable when the selection
 * is changed programmatically by N/P. */
export function selectedStyleFor(feature: Feature): Style[] {
	const ring = (feature.getGeometry() as Polygon).getCoordinates()[0] as Coord[];
	const edges = [0, 1, 2, 3].map((i) => {
		const [x1, y1] = ring[i];
		const [x2, y2] = ring[i + 1];
		return {
			mid: [(x1 + x2) / 2, (y1 + y2) / 2] as Coord,
			length: Math.hypot(x2 - x1, y2 - y1),
			rotation: -Math.atan2(y2 - y1, x2 - x1),
		};
	});
	const long = edges.reduce((a, b) => (b.length > a.length ? b : a));
	const short = edges.reduce((a, b) => (b.length < a.length ? b : a));
	// Concentric and opaque, not a wash. A wide semi-transparent halo blends
	// with whatever orthophoto is underneath — amber over asphalt, over grass,
	// over a blue container — and every one of those blends is a different
	// muddy brown. Opaque strokes stacked widest-first give three crisp bands
	// instead: a dark rim that reads on pale concrete, the amber accent that
	// means "selected", and the class colour as the core.
	//
	// Widths are sized for the object, not the screen: a 3 m vehicle is ~24 px
	// across at z16, so the whole ring has to stay under about a third of that
	// or it swallows the box it is marking.
	//
	// No fill here. `styleFor` already supplies one, and stacking a second was
	// half the muddiness.
	return [
		new Style({ stroke: new Stroke({ color: 'rgba(8,10,14,0.9)', width: 8 }) }),
		new Style({ stroke: new Stroke({ color: '#ffc14d', width: 5 }) }),
		styleFor(feature), // the class colour, as the core of the ring
		edgeLabel(long, `${feature.get('length_m')} m`),
		edgeLabel(short, `${feature.get('width_m')} m`),
	];
}

/** A measurement label at an edge's midpoint, rotated with the edge and
 * flipped the right way up whatever the box's heading. */
function edgeLabel(edge: { mid: Coord; rotation: number }, text: string): Style {
	let rotation = ((edge.rotation % Math.PI) + Math.PI) % Math.PI;
	if (rotation > Math.PI / 2) rotation -= Math.PI;
	return new Style({
		geometry: new Point(edge.mid),
		text: new Text({
			text,
			font: '600 11px ui-monospace, monospace',
			fill: new Fill({ color: '#ffd88a' }),
			backgroundFill: new Fill({ color: 'rgba(5,7,10,0.72)' }),
			padding: [1, 4, 1, 4],
			rotation,
		}),
	});
}
