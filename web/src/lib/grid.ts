/**
 * The Helsinki ETRS-GK25 tile grid, in the browser.
 *
 * Every constant here mirrors `src/rekka_ai/imagery/tiles.py`, which is the
 * tested definition. Keep them identical: if the two drift, labels drawn here
 * land somewhere else when the Python side reads them back.
 */
import proj4 from 'proj4';
import { get as getProjection } from 'ol/proj';
import { register } from 'ol/proj/proj4';
import WMTS from 'ol/source/WMTS';
import WMTSTileGrid from 'ol/tilegrid/WMTS';
import TileLayer from 'ol/layer/Tile';

import { GK25_DEF, TM35FIN_DEF } from '$lib/projection';

export const GRID = 'EPSG:3879';
export const TILE_SIZE = 256;
export const MAX_ZOOM = 17;
export const BASE_RESOLUTION = 8192;

/**
 * Top-left corner of the grid, as [easting, northing].
 *
 * GetCapabilities publishes this as "northing easting" because EPSG:3879
 * declares (north, east) axis order. OpenLayers wants x,y — so the numbers are
 * swapped relative to the capabilities document, deliberately.
 */
export const ORIGIN: [number, number] = [24451424, 8388608];

/**
 * Decimal places kept for a stored coordinate — millimetres, mirroring
 * `geo.COORD_DECIMALS`. Both writers round to the same place, which is what
 * lets an untouched coordinate survive a save as the same digits.
 */
export const COORD_DECIMALS = 3;

/**
 * The `crs` member label files carry, mirroring `geo.crs_member`. Label files
 * are EPSG:3879, not the WGS84 RFC 7946 mandates, and this is what says so.
 */
export const GRID_CRS = {
	type: 'name',
	properties: { name: 'urn:ogc:def:crs:EPSG::3879' },
} as const;

export const ENDPOINT = 'https://kartta.hel.fi/ws/geoserver/avoindata/gwc/service/wmts';
export const MATRIX_SET = 'ETRS-GK25';
export const ATTRIBUTION = '© Helsingin kaupunki, Kaupunkimittauspalvelut';

proj4.defs(GRID, GK25_DEF);
proj4.defs('EPSG:3067', TM35FIN_DEF);
register(proj4);

const span = BASE_RESOLUTION * TILE_SIZE; // ground width of zoom 0, in metres
export const projection = getProjection(GRID)!;
projection.setExtent([ORIGIN[0], ORIGIN[1] - span, ORIGIN[0] + span, ORIGIN[1]]);

export const resolutions = Array.from({ length: MAX_ZOOM + 1 }, (_, z) => BASE_RESOLUTION / 2 ** z);
const matrixIds = resolutions.map((_, z) => `${MATRIX_SET}:${z}`);

export function orthoLayer(layerName: string): TileLayer<WMTS> {
	return new TileLayer({
		source: new WMTS({
			url: ENDPOINT,
			layer: layerName,
			matrixSet: MATRIX_SET,
			format: 'image/jpeg',
			style: 'raster',
			projection,
			attributions: ATTRIBUTION,
			tileGrid: new WMTSTileGrid({
				origin: ORIGIN,
				resolutions,
				matrixIds,
				tileSize: TILE_SIZE,
			}),
			wrapX: false,
		}),
	});
}

/** Metres per pixel at a zoom level — used to show scale while drawing. */
export function resolutionAt(zoom: number): number {
	return BASE_RESOLUTION / 2 ** zoom;
}
