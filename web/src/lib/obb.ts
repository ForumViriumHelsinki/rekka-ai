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
/**
 * Rotation is only ever a correction: a hand-drawn box takes its heading from
 * the nose-to-tail click, and a detected one arrives roughly right. Nothing
 * needs to be spun a long way, so both steps are sized for the last degree
 * rather than the first ninety.
 *
 * 1° was too coarse because it is not a small angle at this scale: on a 16 m
 * box it swings each end through 14 cm, which is a whole pixel of misalignment
 * at both ends at once at z16 — and the map zooms to ~0.8 cm/px, where the
 * same press jumps an end by 18 px. 0.2° moves that end 2.8 cm.
 *
 * Not finer than 0.2°, and not an odd fraction of it: `heading_deg` is stored
 * rounded to one decimal, so 0.1° is the smallest change that can survive a
 * save. A step below that would leave the operator pressing a key and watching
 * nothing happen.
 */
export const ROTATE_STEP_DEG = 0.2;
export const ROTATE_COARSE_DEG = 1;

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
/**
 * The box's axis as [nose, tail]: the midpoints of its two short edges.
 *
 * Ordered by geometry, and deliberately *not* by position in the ring. Reading the
 * pair off in ring order looks equivalent and is not: `boxFromCentreline` lays
 * a box down as `[nose+h, tail+h, tail-h, nose-h]`, so its first short edge is
 * the one at the **tail**, and taking that as the nose hands back a reversed
 * axis. Rebuilding from a reversed axis returns the same box with its vertices
 * rotated by two — geometrically identical, so nothing looked wrong — and the
 * next read reverses it again. Every arrow-key press therefore flipped the
 * ring between two orderings, and anything keyed to a ring index went with it:
 * the measurement labels jumped from one side of the box to the other and
 * back, on rotation and on length alike.
 *
 * Nose and tail are interchangeable here on purpose — a parked vehicle's axis
 * has no direction, which is why `measure` reports heading modulo 180 — so
 * fixing the order costs nothing and buys a ring that stops moving.
 *
 * The ordering keys on a diagonal rather than on an axis. Ranking the two ends
 * by northing ties whenever the box runs east–west, and by easting whenever it
 * runs north–south, and those are precisely the headings vehicles park at:
 * beside a kerb, square to a building. On a tie the comparison falls through to
 * float noise in the last bits, which flips at random. `x + y` only ties for a
 * box on the other diagonal, and the `x - y` fallback settles that.
 *
 * Picking one end of an axis cannot be continuous through a whole revolution —
 * the box maps onto itself every 180°, so the ends must swap somewhere. This
 * only chooses *where*, and puts it somewhere the operator rarely works.
 */
export function centrelineOf(ring: Coord[]): [Coord, Coord] {
	const edge = (i: number) => Math.hypot(ring[i + 1][0] - ring[i][0], ring[i + 1][1] - ring[i][1]);
	const short = edge(0) <= edge(1) ? 0 : 1;
	const mid = (i: number): Coord => [
		(ring[i][0] + ring[i + 1][0]) / 2,
		(ring[i][1] + ring[i + 1][1]) / 2,
	];
	const a = mid(short);
	const b = mid((short + 2) % 4);
	const key = (p: Coord) => p[0] + p[1];
	const tie = (p: Coord) => p[0] - p[1];
	// Sub-millimetre: below the precision a coordinate is ever stored at, so a
	// difference smaller than this is noise rather than geometry.
	const first = Math.abs(key(a) - key(b)) > 1e-6 ? key(a) > key(b) : tie(a) > tie(b);
	return first ? [a, b] : [b, a];
}

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

/**
 * Move one end of a centreline along its own axis, leaving the other end put.
 *
 * The handle drag's edit: the operator has one end of the box right and wants
 * the other one somewhere else, and midpoint scaling (`scaleCentreline`)
 * would move the end that was already right. Which end moves comes from the
 * pointer, never from the nose/tail ordering — that ordering is a geometric
 * convention (see `centrelineOf`), and the operator has no way to know it.
 *
 * Only the on-axis part of the drag is applied. The perpendicular part is
 * dropped rather than followed: the gesture changes length, not heading —
 * a free drag would quietly rotate about the anchored end, and rotation
 * already has the arrow keys.
 *
 * `width` floors the result for the same reason as `scaleCentreline`: under
 * its own width the box swaps length and width in `measure` and jumps its
 * heading by 90°. Dragging back past the anchored end therefore stops at the
 * floor instead of flipping the box inside out.
 */
export function moveEnd(
	nose: Coord,
	tail: Coord,
	end: 'nose' | 'tail',
	target: Coord,
	width = MIN_LENGTH_M,
): [Coord, Coord] {
	const [anchor, moving] = end === 'nose' ? [tail, nose] : [nose, tail];
	const dx = moving[0] - anchor[0];
	const dy = moving[1] - anchor[1];
	const length = Math.hypot(dx, dy);
	// A degenerate centreline has no axis to extend along; leave it alone.
	if (length < 1e-6) return [nose, tail];
	const ux = dx / length;
	const uy = dy / length;
	// Projection onto the axis. Negative means the pointer went back past the
	// anchor, which the floor turns into "as short as it goes", not a flip.
	const along = (target[0] - anchor[0]) * ux + (target[1] - anchor[1]) * uy;
	const next = Math.max(clampLength(along), width);
	const moved: Coord = [anchor[0] + ux * next, anchor[1] + uy * next];
	return end === 'nose' ? [moved, anchor] : [anchor, moved];
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
