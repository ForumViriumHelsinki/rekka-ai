/**
 * One step back, for the edit the operator did not mean to make.
 *
 * Every change here is autosaved within a second, so there is no moment where
 * a mistake is still uncommitted and a mis-drag is written to `labels/` before
 * it is even noticed. Git catches it afterwards — the files are versioned and,
 * since they stopped churning, a stray move shows as a handful of changed
 * coordinates — but "review the diff at the end of the session" is not an
 * answer to "I just nudged that box off a roof".
 *
 * Entries record what a box *was*, not what changed, because restoring a
 * remembered state is exact where replaying an inverse is not: rotation and
 * scaling both round, and an inverse nudge would leave the box a few
 * millimetres from where it started while looking like it had gone home.
 *
 * A burst of arrow presses collapses into one entry. Holding a key emits an
 * edit per repeat, and undoing sixty of those one at a time is not undo — the
 * operator means "put it back where it was before I started nudging".
 */
import type Feature from 'ol/Feature';
import type Polygon from 'ol/geom/Polygon';
import type VectorSource from 'ol/source/Vector';

import type { Coord } from '$lib/obb';

/** Consecutive edits to one box within this window count as one action. */
export const COALESCE_MS = 700;
/** Deep enough to cover a wrong turn, shallow enough to stay a list. */
export const UNDO_LIMIT = 100;

type Entry =
	| { kind: 'edit'; at: number; feature: Feature; ring: Coord[]; properties: Properties }
	| { kind: 'add'; at: number; feature: Feature }
	| { kind: 'remove'; at: number; feature: Feature; index: number };

type Properties = Record<string, unknown>;

function ringOf(feature: Feature): Coord[] {
	return (feature.getGeometry() as Polygon).getCoordinates()[0].map((c) => [...c] as Coord);
}

function propertiesOf(feature: Feature): Properties {
	const { geometry: _geometry, ...rest } = feature.getProperties();
	return { ...rest };
}

export class UndoStack {
	#entries = $state<Entry[]>([]);
	#now: () => number;

	/** `now` is injectable so the coalescing window can be tested without
	 * waiting out real time. */
	constructor(now: () => number = Date.now) {
		this.#now = now;
	}

	get depth(): number {
		return this.#entries.length;
	}

	/** Dropped on every area change: the features these entries point at are
	 * gone from the source, and restoring one would resurrect another area's
	 * box under this area's name. */
	clear(): void {
		this.#entries = [];
	}

	/** Call *before* changing a box's geometry or verdict. */
	edit(feature: Feature): void {
		const last = this.#entries[this.#entries.length - 1];
		// Mid-burst on the same box: the state already remembered is the one
		// worth going back to, so keep it and let this change fold into it.
		if (last?.kind === 'edit' && last.feature === feature && this.#now() - last.at < COALESCE_MS) {
			last.at = this.#now();
			return;
		}
		this.#push({
			kind: 'edit',
			at: this.#now(),
			feature,
			ring: ringOf(feature),
			properties: propertiesOf(feature),
		});
	}

	/** Call after a hand-drawn box joins the source. */
	added(feature: Feature): void {
		this.#push({ kind: 'add', at: this.#now(), feature });
	}

	/** Call *before* removing a box, while it is still in the source. */
	removed(feature: Feature, index: number): void {
		this.#push({ kind: 'remove', at: this.#now(), feature, index });
	}

	/**
	 * Undo the last change. Returns the box to leave selected — the one just
	 * restored, so the operator can see what came back — or `null` when the
	 * change undone was an addition and there is nothing to point at.
	 */
	undo(source: VectorSource): Feature | null {
		const entry = this.#entries.pop();
		if (!entry) return null;
		if (entry.kind === 'add') {
			source.removeFeature(entry.feature);
			return null;
		}
		if (entry.kind === 'remove') {
			source.addFeature(entry.feature);
			return entry.feature;
		}
		(entry.feature.getGeometry() as Polygon).setCoordinates([entry.ring]);
		entry.feature.setProperties(entry.properties);
		return entry.feature;
	}

	#push(entry: Entry): void {
		this.#entries.push(entry);
		if (this.#entries.length > UNDO_LIMIT) this.#entries.shift();
	}
}
