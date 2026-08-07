/**
 * Draw mode: placing a new box by hand.
 *
 * Drawing is a three-phase state machine, not an OL Draw interaction:
 * 0 = click the nose, 1 = click the tail, 2 = scroll for width, click or
 * Enter to place. The OL Draw interaction cannot wait for a third input,
 * which is why the wheel used to fight the second click.
 *
 * The class owns the gesture — the phase, the in-progress axis, the sketch
 * preview and its listeners — while the page owns what a committed box
 * *means*: the label source, undo, selection and the dirty flag all arrive
 * through `onCommit`. `drawing` and `phase` are rune state because the
 * footer, the cursor and the keymap all read them.
 *
 * One box per start: commit or stop tears the gesture down, so a stray click
 * can never place an unintended box.
 */
import type Map from 'ol/Map';
import Feature from 'ol/Feature';
import LineString from 'ol/geom/LineString';
import Polygon from 'ol/geom/Polygon';
import type Select from 'ol/interaction/Select';
import type Translate from 'ol/interaction/Translate';
import type MouseWheelZoom from 'ol/interaction/MouseWheelZoom';
import type VectorSource from 'ol/source/Vector';
import type { EventsKey } from 'ol/events';
import { unByKey } from 'ol/Observable';

import { boxFromCentreline, type Coord } from '$lib/obb';

type Deps = {
	map: Map;
	/** The preview lives in its own source so it is never saved. */
	sketchSource: VectorSource;
	/** The interactions the gesture takes over: the wheel sets width while
	 * drawing, and a draw click must not select or move a box. */
	select: Select;
	translate: Translate;
	wheelZoom: MouseWheelZoom;
	/** The width the next box starts from — the page's state, nudged by the
	 * wheel while drawing. */
	getWidth: () => number;
	/** Turn the committed axis into a real feature. */
	onCommit: (nose: Coord, tail: Coord) => void;
};

export class DrawMode {
	drawing = $state(false);
	phase = $state<0 | 1 | 2>(0);
	private nose: Coord | null = null;
	private tail: Coord | null = null;
	private sketch: Feature | null = null;
	private listeners: EventsKey[] = [];

	constructor(private readonly deps: Deps) {}

	start() {
		if (this.drawing) return;
		this.drawing = true;
		this.cancel(); // drop whatever a previous, aborted draw left behind
		// Reactivated in stop(); take them in the same order there.
		this.deps.wheelZoom.setActive(false);
		this.deps.select.setActive(false);
		this.deps.translate.setActive(false);
		this.listeners.push(
			this.deps.map.on('click', this.onClick),
			this.deps.map.on('pointermove', this.onMove),
		);
	}

	stop() {
		if (!this.drawing) return;
		unByKey(this.listeners);
		this.listeners = [];
		this.cancel();
		this.drawing = false;
		this.deps.wheelZoom.setActive(true);
		this.deps.select.setActive(true);
		this.deps.translate.setActive(true);
	}

	/** Esc mid-draw: drop the in-progress axis, stay in draw mode. */
	cancel() {
		this.nose = null;
		this.tail = null;
		this.phase = 0;
		this.clearSketch();
	}

	commit() {
		if (!this.nose || !this.tail) return;
		this.deps.onCommit(this.nose, this.tail);
		this.stop();
	}

	/** Re-preview the width phase after the wheel changed the width. Only the
	 * width phase previews live — before the tail exists a box would just be
	 * noise chasing the cursor. */
	preview() {
		if (this.phase === 2 && this.nose && this.tail)
			this.showSketch(new Polygon([boxFromCentreline(this.nose, this.tail, this.deps.getWidth())]));
	}

	destroy() {
		unByKey(this.listeners);
		this.listeners = [];
	}

	private onClick = (event: { coordinate: number[] }) => {
		const at = event.coordinate as Coord;
		if (this.phase === 0) {
			this.nose = at;
			this.phase = 1;
		} else if (this.phase === 1) {
			this.tail = at;
			this.phase = 2;
			this.preview();
		} else {
			this.commit();
		}
	};

	private onMove = (event: { coordinate: number[] }) => {
		if (this.phase === 1 && this.nose) {
			// A plain centreline — the box only appears once the tail is set,
			// so the width step reads as width, not as a moving rectangle.
			this.showSketch(new LineString([this.nose, event.coordinate as Coord]));
		}
	};

	private showSketch(geometry: LineString | Polygon) {
		if (this.sketch) this.sketch.setGeometry(geometry);
		else {
			this.sketch = new Feature(geometry);
			this.deps.sketchSource.addFeature(this.sketch);
		}
	}

	private clearSketch() {
		if (this.sketch) {
			this.deps.sketchSource.removeFeature(this.sketch);
			this.sketch = null;
		}
	}
}
