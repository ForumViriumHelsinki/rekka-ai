/**
 * Reading and writing the repository's label files.
 *
 * The tool is a local editor over `labels/` — git is the version control, so
 * there is no database and no auth. It only ever runs on localhost.
 */
import { readFile, writeFile, rename, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { load } from 'js-yaml';
import proj4 from 'proj4';

import { GK25_DEF, TM35FIN_DEF } from '$lib/projection';

/** Repository root — the app lives in `web/`, labels live beside it. */
export const ROOT = resolve(process.cwd(), '..');
export const LABELS_DIR = resolve(ROOT, 'labels');
export const AOI_FILE = resolve(ROOT, 'aois/helsinki.yaml');

export { LATEST_LAYER } from '$lib/schema';

export type Aoi = {
	name: string;
	role: string;
	split: string;
	notes: string;
	/** Extent in EPSG:3879, as [minX, minY, maxX, maxY]. */
	extent: [number, number, number, number];
};

type RawAoi = {
	name: string;
	bbox: [number, number, number, number];
	role?: string;
	split?: string;
	notes?: string;
};

/** Reject anything that could escape the labels directory. */
function labelPath(aoi: string): string {
	if (!/^[A-Za-z0-9_-]+$/.test(aoi)) throw new Error(`bad AOI name: ${aoi}`);
	return resolve(LABELS_DIR, `${aoi}.geojson`);
}

export async function listAois(): Promise<Aoi[]> {
	const doc = load(await readFile(AOI_FILE, 'utf8')) as {
		crs: string;
		aois: RawAoi[];
	};
	// The collection declares its own CRS; project every corner, because a box
	// in EPSG:3067 is not axis-aligned in EPSG:3879 and two corners under-cover
	// the true extent.
	const from = doc.crs === 'EPSG:3067' ? TM35FIN_DEF : GK25_DEF;
	return doc.aois.map((a) => {
		const [x0, y0, x1, y1] = a.bbox;
		const corners = [
			[x0, y0],
			[x1, y0],
			[x1, y1],
			[x0, y1],
		].map(([x, y]) => proj4(from, GK25_DEF, [x, y]) as [number, number]);
		const xs = corners.map((c) => c[0]);
		const ys = corners.map((c) => c[1]);
		return {
			name: a.name,
			role: a.role ?? 'positive',
			split: a.split ?? 'train',
			notes: a.notes ?? '',
			extent: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
		};
	});
}

export async function readLabels(aoi: string): Promise<unknown> {
	const path = labelPath(aoi);
	if (!existsSync(path)) return { type: 'FeatureCollection', features: [] };
	return JSON.parse(await readFile(path, 'utf8'));
}

let writeCounter = 0;

export async function writeLabels(aoi: string, collection: unknown): Promise<void> {
	const path = labelPath(aoi);
	// Write to a temporary sibling and rename, so a crash mid-write cannot
	// truncate hours of labelling into an unparseable file.
	//
	// The temporary name is unique per write, not a fixed `.tmp`: autosave and
	// a manual Ctrl+S can overlap, as can two tabs, and a shared scratch file
	// lets two writers interleave into one and rename the mixture over the
	// labels. This is the file the project cannot afford to corrupt.
	const temporary = `${path}.${process.pid}.${Date.now()}.${writeCounter++}.tmp`;
	try {
		await writeFile(temporary, JSON.stringify(collection, null, 2) + '\n', 'utf8');
		await rename(temporary, path);
	} catch (e) {
		// A failed write must not leave scratch files beside the labels.
		await rm(temporary, { force: true });
		throw e;
	}
}

/** Review counts for one area, so the sidebar can show where work remains. */
export async function counts(aoi: string): Promise<{ total: number; reviewed: number }> {
	const collection = (await readLabels(aoi)) as {
		features?: { properties?: { status?: string } }[];
	};
	const features = collection.features ?? [];
	return {
		total: features.length,
		reviewed: features.filter((f) => (f.properties?.status ?? 'candidate') !== 'candidate').length,
	};
}

export async function labelledAois(): Promise<string[]> {
	if (!existsSync(LABELS_DIR)) return [];
	return (await readdir(LABELS_DIR))
		.filter((f) => f.endsWith('.geojson'))
		.map((f) => f.replace(/\.geojson$/, ''));
}
