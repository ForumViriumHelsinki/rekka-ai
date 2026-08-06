import { json, error } from '@sveltejs/kit';
import { readLabels, writeLabels } from '$lib/server/store';
import { GRID_CRS } from '$lib/grid';
import type { RequestEvent } from '@sveltejs/kit';

/** A bad AOI name is the caller's fault (400); anything else is ours (500). */
function statusOf(e: unknown): number {
	return e instanceof Error && e.message.startsWith('bad AOI name') ? 400 : 500;
}

export async function GET({ params }: RequestEvent) {
	try {
		return json(await readLabels(params.aoi!));
	} catch (e) {
		throw error(statusOf(e), String(e));
	}
}

export async function PUT({ params, request }: RequestEvent) {
	const body = await request.json();
	if (body?.type !== 'FeatureCollection' || !Array.isArray(body.features)) {
		throw error(400, 'expected a GeoJSON FeatureCollection');
	}
	// Label files are EPSG:3879 and say so. A body without the member — a tab
	// left open across a deploy is the plausible one — would write coordinates
	// the Python side then refuses to read, over labels that cannot be
	// regenerated. Cheaper to reject the save and let it retry.
	if (body.crs?.properties?.name !== GRID_CRS.properties.name) {
		throw error(400, `expected a collection declaring ${GRID_CRS.properties.name}`);
	}
	try {
		await writeLabels(params.aoi!, body);
	} catch (e) {
		throw error(statusOf(e), String(e));
	}
	return json({ saved: body.features.length });
}
