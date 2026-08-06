import { json, error } from '@sveltejs/kit';
import { readLabels, writeLabels } from '$lib/server/store';
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
	try {
		await writeLabels(params.aoi!, body);
	} catch (e) {
		throw error(statusOf(e), String(e));
	}
	return json({ saved: body.features.length });
}
