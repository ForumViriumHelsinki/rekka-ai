/**
 * proj4 definitions for the two Finnish projected CRSs, shared by the client
 * tile grid and the server store so each string exists exactly once. The
 * Python side carries the same CRSs in `src/rekka_ai/geo.py`.
 */
export const TM35FIN_DEF =
	'+proj=utm +zone=35 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs';
export const GK25_DEF =
	'+proj=tmerc +lat_0=0 +lon_0=25 +k=1 +x_0=25500000 +y_0=0 ' +
	'+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs';
