import { expect, test } from 'vitest';

import {
	boxFromCentreline,
	measure,
	clampWidth,
	scaleCentreline,
	rotateCentreline,
	centrelineOf,
	ROTATE_STEP_DEG,
	ROTATE_COARSE_DEG,
	type Coord,
} from './obb';

const ok = (name: string, cond: boolean, got?: unknown) =>
	test(name, () => expect(cond, String(got ?? '')).toBe(true));
const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) < tol;

// A 16 m nose-to-tail centreline, 3 m wide, pointing east.
const nose: Coord = [25496000, 6673000];
const tail: Coord = [25496016, 6673000];
const ring = boxFromCentreline(nose, tail, 3);
ok('ring is closed', ring.length === 5 && ring[0][0] === ring[4][0] && ring[0][1] === ring[4][1]);
const m = measure(ring);
ok('length is the centreline length', near(m.length, 16), m.length);
ok('width is what was asked for', near(m.width, 3), m.width);
ok('east-pointing axis reads as bearing 90', near(m.heading, 90, 1e-9), m.heading);

// North-pointing
const north = measure(boxFromCentreline([0, 0], [0, 16], 3));
ok('north-pointing axis reads as bearing 0', near(north.heading, 0, 1e-9), north.heading);

// Rotation invariance: length/width must not depend on heading.
let invariant = true;
for (let deg = 0; deg < 360; deg += 17) {
	const r = (deg * Math.PI) / 180;
	const t: Coord = [nose[0] + 16 * Math.cos(r), nose[1] + 16 * Math.sin(r)];
	const mm = measure(boxFromCentreline(nose, t, 2.5));
	if (!near(mm.length, 16, 1e-6) || !near(mm.width, 2.5, 1e-6)) invariant = false;
}
ok('length/width invariant across 360 degrees', invariant);

// Heading is modulo 180 - a parked vehicle's axis has no direction.
const fwd = measure(boxFromCentreline([0, 0], [10, 10], 3)).heading;
const rev = measure(boxFromCentreline([10, 10], [0, 0], 3)).heading;
ok('reversing nose and tail gives the same bearing', near(fwd, rev, 1e-9), `${fwd} vs ${rev}`);

// The box must be centred on the centreline, not offset to one side.
const cx = ring.slice(0, 4).reduce((s, c) => s + c[0], 0) / 4;
const cy = ring.slice(0, 4).reduce((s, c) => s + c[1], 0) / 4;
ok('box is centred on the centreline', near(cx, 25496008) && near(cy, 6673000), [cx, cy]);

// Degenerate input (before the second click) must not produce NaN.
const degenerate = boxFromCentreline(nose, nose, 3);
ok(
	'zero-length sketch is finite',
	degenerate.every((c) => Number.isFinite(c[0]) && Number.isFinite(c[1])),
);

ok('width clamps low', clampWidth(0.2) === 1.5);
ok('width clamps high', clampWidth(99) === 6.0);

// --- length and rotation nudges ---------------------------------------------
// Both edit the centreline, so the ring they rebuild stays a rectangle.

const centre = (r: Coord[]): Coord => [
	r.slice(0, 4).reduce((s, c) => s + c[0], 0) / 4,
	r.slice(0, 4).reduce((s, c) => s + c[1], 0) / 4,
];

const [longNose, longTail] = scaleCentreline(nose, tail, 2);
const longer = measure(boxFromCentreline(longNose, longTail, 3));
ok('length grows by the step', near(longer.length, 18), longer.length);
ok('growing keeps width', near(longer.width, 3), longer.width);
ok('growing keeps heading', near(longer.heading, 90, 1e-9), longer.heading);
ok(
	'growing keeps the centre put',
	near(centre(boxFromCentreline(longNose, longTail, 3))[0], 25496008),
	centre(boxFromCentreline(longNose, longTail, 3)),
);

// Shrinking stops at the box's own width: `measure` calls the longer side the
// length, so going under would swap length and width and jump the heading 90.
const [shortNose, shortTail] = scaleCentreline(nose, tail, -100, 3);
const shrunk = measure(boxFromCentreline(shortNose, shortTail, 3));
ok('length never shrinks under the box width', near(shrunk.length, 3), shrunk.length);
ok('shrinking to the floor keeps width intact', near(shrunk.width, 3), shrunk.width);
ok(
	'length clamps low without a width',
	near(measure(boxFromCentreline(...scaleCentreline(nose, tail, -100), 1.5)).length, 2),
);
const [bigNose, bigTail] = scaleCentreline(nose, tail, 999);
ok('length clamps high', near(measure(boxFromCentreline(bigNose, bigTail, 3)).length, 30));

// A degenerate centreline has no axis, so scaling must leave it alone.
const [dn, dt] = scaleCentreline(nose, nose, 5);
ok('scaling a zero-length centreline is a no-op', dn[0] === nose[0] && dt[0] === nose[0]);

const [rotNose, rotTail] = rotateCentreline(nose, tail, 30);
const rotated = measure(boxFromCentreline(rotNose, rotTail, 3));
// 1e-6 rather than 1e-9: EPSG:3879 eastings run to 2.5e7, where a double has
// only about 4e-9 m of precision left. Still far inside the 0.5 m tolerance
// labels.validate applies to the rectangle itself.
ok('positive rotation turns clockwise', near(rotated.heading, 120, 1e-6), rotated.heading);
ok('rotating preserves length', near(rotated.length, 16), rotated.length);
ok(
	'rotating keeps the centre put',
	near(centre(boxFromCentreline(rotNose, rotTail, 3))[0], 25496008) &&
		near(centre(boxFromCentreline(rotNose, rotTail, 3))[1], 6673000),
);

const [backNose, backTail] = rotateCentreline(rotNose, rotTail, -30);
ok(
	'rotating back returns the original axis',
	near(backNose[0], nose[0], 1e-6) && near(backTail[0], tail[0], 1e-6),
	[backNose, backTail],
);

// The invariant the whole approach exists to protect: whatever the nudges do,
// the ring stays a rectangle, because labels.validate rejects anything else.
let stillRectangular = true;
let [pn, pt] = [nose, tail];
for (let i = 0; i < 40; i++) {
	[pn, pt] = rotateCentreline(...scaleCentreline(pn, pt, i % 2 ? -0.25 : 1.0), 7);
	const r = boxFromCentreline(pn, pt, 3);
	const sides = [0, 1, 2, 3].map((k) =>
		Math.hypot(r[(k + 1) % 4][0] - r[k][0], r[(k + 1) % 4][1] - r[k][1]),
	);
	const diag = [
		Math.hypot(r[2][0] - r[0][0], r[2][1] - r[0][1]),
		Math.hypot(r[3][0] - r[1][0], r[3][1] - r[1][1]),
	];
	if (
		!near(sides[0], sides[2], 1e-6) ||
		!near(sides[1], sides[3], 1e-6) ||
		!near(diag[0], diag[1], 1e-6)
	)
		stillRectangular = false;
}
ok('40 mixed nudges leave the ring a rectangle', stillRectangular);

// `heading_deg` is stored rounded to one decimal, so a step under 0.1 deg can
// round away entirely: the operator presses an arrow and the box does not
// move. The fine step must survive that rounding, and coarse must outrank it.
ok(
	'the fine rotation step survives heading_deg rounding',
	Math.round(ROTATE_STEP_DEG * 10) / 10 === ROTATE_STEP_DEG && ROTATE_STEP_DEG >= 0.1,
	ROTATE_STEP_DEG,
);
ok('coarse rotation outranks fine', ROTATE_COARSE_DEG > ROTATE_STEP_DEG);

// `centrelineOf` used to read its ends off in ring order, which for a box laid
// down by `boxFromCentreline` hands back [tail, nose]. Rebuilding from a
// reversed axis returns the same box with its vertices rotated by two, and the
// next read reverses it again — so the ring flipped between two orderings on
// every arrow press. Canonical ordering makes the rebuild a fixed point.
{
	const once = boxFromCentreline(...centrelineOf(boxFromCentreline(nose, tail, 3) as Coord[]), 3);
	const twice = boxFromCentreline(...centrelineOf(once as Coord[]), 3);
	ok(
		'rebuilding from the recovered centreline is a fixed point',
		JSON.stringify(once) === JSON.stringify(twice),
	);
}
