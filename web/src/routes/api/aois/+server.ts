import { json } from '@sveltejs/kit';
import { listAois, labelledAois, counts, LATEST_LAYER } from '$lib/server/store';

export async function GET() {
	const [aois, staged] = await Promise.all([listAois(), labelledAois()]);
	const have = new Set(staged);
	return json({
		layer: LATEST_LAYER,
		aois: await Promise.all(
			aois.map(async (a) => ({
				...a,
				staged: have.has(a.name),
				...(have.has(a.name) ? await counts(a.name) : { total: 0, reviewed: 0 }),
			})),
		),
	});
}
