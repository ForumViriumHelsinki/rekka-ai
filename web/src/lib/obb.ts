/**
 * Oriented boxes from a centreline.
 *
 * Measured over 392 bootstrap candidates, vehicle width has a standard
 * deviation of 0.37 m while length varies by 4.32 m — trucks are road-legal
 * width by law. So the operator draws the axis that actually varies (nose to
 * tail) and width stays a default they can nudge, rather than placing four
 * corners for every vehicle.
 */

/** Default half-metre-rounded width, near the median of real detections. */
export const DEFAULT_WIDTH_M = 3.0;
export const MIN_WIDTH_M = 1.5;
export const MAX_WIDTH_M = 6.0;
export const WIDTH_STEP_M = 0.1;

export type Coord = [number, number];

/**
 * The four corners of the box around a centreline, closed for GeoJSON.
 * Coordinates are EPSG:3879 metres, so `width` is metres too.
 */
export function boxFromCentreline(nose: Coord, tail: Coord, width: number): Coord[] {
	const dx = tail[0] - nose[0];
	const dy = tail[1] - nose[1];
	const length = Math.hypot(dx, dy);
	// Before the second click the two points coincide; fall back to a
	// north-pointing unit vector so the sketch is visible rather than empty.
	const [ux, uy] = length < 1e-6 ? [0, 1] : [dx / length, dy / length];
	const hx = (-uy * width) / 2;
	const hy = (ux * width) / 2;
	const a: Coord = [nose[0] + hx, nose[1] + hy];
	const b: Coord = [tail[0] + hx, tail[1] + hy];
	const c: Coord = [tail[0] - hx, tail[1] - hy];
	const d: Coord = [nose[0] - hx, nose[1] - hy];
	return [a, b, c, d, a];
}

/** Length, width and compass heading of a ring, in metres and degrees. */
export function measure(ring: Coord[]): {
	length: number;
	width: number;
	heading: number;
} {
	const [p0, p1, p2] = ring;
	const first = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
	const second = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
	const [dEast, dNorth] =
		first >= second ? [p1[0] - p0[0], p1[1] - p0[1]] : [p2[0] - p1[0], p2[1] - p1[1]];
	// Modulo 180: a parked vehicle's axis has no direction, so nose-north and
	// nose-south are the same orientation.
	const heading = ((Math.atan2(dEast, dNorth) * 180) / Math.PI + 180) % 180;
	return { length: Math.max(first, second), width: Math.min(first, second), heading };
}

export function clampWidth(width: number): number {
	return Math.min(MAX_WIDTH_M, Math.max(MIN_WIDTH_M, width));
}

/**
 * Nudges for correcting a box that is close but not right. Length and heading
 * are corrected through the centreline rather than the ring: `labels.validate`
 * rejects any ring that is not a rectangle, so dragging a single corner would
 * produce a label that looks fine on the map and blocks the export hours later.
 * Rebuilding from a centreline makes that mistake unrepresentable.
 */
export const LENGTH_STEP_M = 0.25;
export const LENGTH_COARSE_M = 1.0;
export const MIN_LENGTH_M = 2.0;
export const MAX_LENGTH_M = 30.0;
export const ROTATE_STEP_DEG = 1;
export const ROTATE_COARSE_DEG = 5;

export function clampLength(length: number): number {
	return Math.min(MAX_LENGTH_M, Math.max(MIN_LENGTH_M, length));
}

/**
 * Grow or shrink a centreline about its midpoint, so correcting a length
 * leaves the vehicle centred where the operator already put it.
 *
 * `width` is the box's own width, and is a floor on the result: `measure`
 * calls the *longer* side the length, so a box shrunk under its own width
 * would silently swap its length and width and jump its heading by 90°.
 */
export function scaleCentreline(
	nose: Coord,
	tail: Coord,
	delta: number,
	width = MIN_LENGTH_M,
): [Coord, Coord] {
	const dx = tail[0] - nose[0];
	const dy = tail[1] - nose[1];
	const length = Math.hypot(dx, dy);
	// A degenerate centreline has no axis to grow along; leave it for the
	// operator to redraw rather than inventing a direction.
	if (length < 1e-6) return [nose, tail];
	const half = Math.max(clampLength(length + delta), width) / 2;
	const mx = (nose[0] + tail[0]) / 2;
	const my = (nose[1] + tail[1]) / 2;
	const hx = (dx / length) * half;
	const hy = (dy / length) * half;
	return [
		[mx - hx, my - hy],
		[mx + hx, my + hy],
	];
}

/** Rotate a centreline about its midpoint. Positive degrees turn clockwise, so
 * a right-arrow press raises the compass heading `measure` reports. */
export function rotateCentreline(nose: Coord, tail: Coord, degrees: number): [Coord, Coord] {
	const radians = (degrees * Math.PI) / 180;
	const sin = Math.sin(radians);
	const cos = Math.cos(radians);
	const mx = (nose[0] + tail[0]) / 2;
	const my = (nose[1] + tail[1]) / 2;
	const turn = (p: Coord): Coord => {
		const x = p[0] - mx;
		const y = p[1] - my;
		return [mx + x * cos + y * sin, my - x * sin + y * cos];
	};
	return [turn(nose), turn(tail)];
}
