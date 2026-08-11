/**
 * Values this app shares with the Python pipeline.
 *
 * Every constant here is a *mirror*. Python owns the definition; this file
 * restates it so the browser can use it, and `tests/test_web_constants.py`
 * fails if the two ever disagree. Without that test these are just two
 * hardcoded copies with a comment between them — which is how a labeller ends
 * up stamping last year's layer name into this year's boxes.
 *
 * Mirrors, and where the truth lives:
 *
 * - `CLASSES`      -> `labels.CLASSES`
 * - `SOURCE_ZOOM`  -> `cli.BOOTSTRAP_ZOOM`
 * - `LATEST_LAYER` -> `layers.layer_for_year(layers.LATEST_YEAR)`
 *
 * The tile grid is mirrored the same way in `grid.ts`, against `tiles.py`, and
 * covered by the same test.
 */

/** Annotated classes, in the order that fixes their YOLO indices. */
export const CLASSES = ['truck', 'bus', 'van', 'car'] as const;
export type Klass = (typeof CLASSES)[number];

/** The zoom labels are drawn against, and the one bootstrap runs at. */
export const SOURCE_ZOOM = 16;

/** Newest orthophoto layer. Served via /api/aois so the page has no copy. */
export const LATEST_LAYER = 'Ortoilmakuva_2025_5cm';
