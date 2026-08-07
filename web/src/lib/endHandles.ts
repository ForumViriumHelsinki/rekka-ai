/**
 * Draggable end handles for the selected box — the one-ended length edit.
 *
 * The operator's complaint this answers: arrow-key length nudges scale the box
 * about its midpoint, so correcting a length always moved the end that was
 * already right. Dragging a handle moves only that end (`moveEnd` in obb.ts
 * does the geometry); the other end, the heading and the width stay put.
 *
 * The drag is a raw pointer session on the viewport, not an OL interaction —
 * the same pattern as the page's wheel handler, and for the same reason. A
 * gesture that starts on a handle must never enter OpenLayers' own pointer
 * pipeline, where Translate would move the whole box, DragPan would pan under
 * it, and the trailing click would clear the selection. The pointer event
 * targets a canvas *inside* the viewport, so a capture-phase listener on the
 * viewport runs before anything OL registered.
 *
 * Everything this class needs from the page comes in through the constructor
 * as accessors and callbacks; it owns its listeners and its two handle
 * features, and `destroy` leaves nothing behind.
 */
import type Map from 'ol/Map';
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';
import type Polygon from 'ol/geom/Polygon';
import type VectorLayer from 'ol/layer/Vector';
import type VectorSource from 'ol/source/Vector';
import type { EventsKey } from 'ol/events';
import { unByKey } from 'ol/Observable';

import { boxFromCentreline, centrelineOf, measure, moveEnd, type Coord } from '$lib/obb';

type Deps = {
	map: Map;
	/** Topmost layer holding exactly the two handle points, so hit-testing
	 * finds a handle before anything else. */
	layer: VectorLayer<VectorSource>;
	getSelected: () => Feature | null;
	isDrawing: () => boolean;
	/** One undo snapshot, taken while the box is still where it was, so a
	 * single undo reverts the whole drag. */
	onGestureStart: (feature: Feature) => void;
	/** Live feedback during the drag: measurements and the footer text. */
	onGestureMove: (feature: Feature) => void;
	/** The gesture ended; persist it. */
	onGestureEnd: () => void;
};

export class EndHandles {
	private noseHandle: Feature<Point> | null = null;
	private tailHandle: Feature<Point> | null = null;
	private draggingEnd: 'nose' | 'tail' | null = null;
	/** The click that ends (or starts and ends) a handle drag lands wherever
	 * the pointer was released — outside the box, OL would dispatch it as a
	 * map click and Select would drop the box being edited. Swallow one. */
	private swallowNextClick = false;
	private readonly viewport: HTMLElement;
	private readonly mapListeners: EventsKey[] = [];

	constructor(private readonly deps: Deps) {
		const viewport = deps.map.getViewport();
		this.viewport = viewport;
		viewport.addEventListener('pointerdown', this.onDown, { capture: true });
		viewport.addEventListener('pointermove', this.onMove);
		viewport.addEventListener('pointerup', this.onUp);
		viewport.addEventListener('pointercancel', this.onUp);
		viewport.addEventListener('click', this.onSwallow, { capture: true });

		// Grab cursor over a handle, when idle. The drag itself keeps its own
		// cursor; drawing keeps its crosshair.
		this.mapListeners.push(
			deps.map.on('pointermove', (event) => {
				if (this.draggingEnd || deps.isDrawing() || !deps.getSelected()) return;
				const over = deps.map.hasFeatureAtPixel(event.pixel, {
					hitTolerance: 6,
					layerFilter: (layer) => layer === deps.layer,
				});
				viewport.style.cursor = over ? 'grab' : '';
			}),
			// The handles' outward offset is measured in pixels, so a zoom change
			// has to re-lay them out even when the box itself has not moved.
			deps.map.on('moveend', () => this.sync()),
		);
	}

	/** Put the handles outside the selected box's short edges, or drop them.
	 * Called after every change that could move the box: selection, reshape,
	 * resize, translate, undo, zoom.
	 *
	 * The handles sit a fixed screen distance *outside* the box along its axis
	 * rather than on the edge midpoints: the width label lives at exactly that
	 * midpoint (see selectedStyleFor), and a handle on top of it would hide
	 * the measurement it is there to correct. The offset is visual only — the
	 * drag resolves against the real centreline, not the handle position. */
	sync() {
		const source = this.deps.layer.getSource();
		const selected = this.deps.getSelected();
		if (!source) return;
		if (!selected) {
			if (this.noseHandle) source.removeFeature(this.noseHandle);
			if (this.tailHandle) source.removeFeature(this.tailHandle);
			this.noseHandle = this.tailHandle = null;
			return;
		}
		const ring = (selected.getGeometry() as Polygon).getCoordinates()[0] as Coord[];
		const [n, t] = centrelineOf(ring);
		const res = this.deps.map.getView().getResolution() ?? 1;
		const dx = t[0] - n[0];
		const dy = t[1] - n[1];
		const axis = Math.hypot(dx, dy) || 1;
		const out = 12 * res; // ~12 px beyond each end
		const np: Coord = [n[0] - (dx / axis) * out, n[1] - (dy / axis) * out];
		const tp: Coord = [t[0] + (dx / axis) * out, t[1] + (dy / axis) * out];
		if (!this.noseHandle || !this.tailHandle) {
			this.noseHandle = new Feature(new Point(np));
			this.noseHandle.set('end', 'nose');
			this.tailHandle = new Feature(new Point(tp));
			this.tailHandle.set('end', 'tail');
			source.addFeatures([this.noseHandle, this.tailHandle]);
		} else {
			this.noseHandle.getGeometry()?.setCoordinates(np);
			this.tailHandle.getGeometry()?.setCoordinates(tp);
		}
	}

	destroy() {
		const viewport = this.viewport;
		viewport.removeEventListener('pointerdown', this.onDown, { capture: true });
		viewport.removeEventListener('pointermove', this.onMove);
		viewport.removeEventListener('pointerup', this.onUp);
		viewport.removeEventListener('pointercancel', this.onUp);
		viewport.removeEventListener('click', this.onSwallow, { capture: true });
		unByKey(this.mapListeners);
	}

	private onDown = (event: PointerEvent) => {
		const selected = this.deps.getSelected();
		if (event.button !== 0 || this.deps.isDrawing() || !selected) return;
		const hit = this.deps.map.forEachFeatureAtPixel(
			this.deps.map.getEventPixel(event),
			(feature, layer) => (layer === this.deps.layer ? (feature as Feature) : undefined),
			{ hitTolerance: 6 },
		);
		if (!hit) return;
		// From here the gesture is ours; OL never sees the pointerdown.
		event.stopPropagation();
		this.draggingEnd = hit.get('end') as 'nose' | 'tail';
		this.deps.onGestureStart(selected);
		this.viewport.setPointerCapture(event.pointerId);
		this.viewport.style.cursor = 'grabbing';
	};

	private onMove = (event: PointerEvent) => {
		const selected = this.deps.getSelected();
		if (!this.draggingEnd || !selected) return;
		const geometry = selected.getGeometry() as Polygon;
		const ring = geometry.getCoordinates()[0] as Coord[];
		const [nose, tail] = centrelineOf(ring);
		const boxWidth = measure(ring).width;
		// moveEnd keeps the other end put and drops the off-axis part of the
		// drag — the gesture changes length, never heading.
		const [nextNose, nextTail] = moveEnd(
			nose,
			tail,
			this.draggingEnd,
			this.deps.map.getEventCoordinate(event) as Coord,
			boxWidth,
		);
		geometry.setCoordinates([boxFromCentreline(nextNose, nextTail, boxWidth)]);
		this.deps.onGestureMove(selected);
		this.sync();
	};

	private onUp = (event: PointerEvent) => {
		if (!this.draggingEnd) return;
		this.draggingEnd = null;
		this.swallowNextClick = true;
		if (this.viewport.hasPointerCapture(event.pointerId))
			this.viewport.releasePointerCapture(event.pointerId);
		this.viewport.style.cursor = '';
		this.deps.onGestureEnd();
	};

	private onSwallow = (event: MouseEvent) => {
		if (!this.swallowNextClick) return;
		this.swallowNextClick = false;
		event.stopPropagation();
	};
}
