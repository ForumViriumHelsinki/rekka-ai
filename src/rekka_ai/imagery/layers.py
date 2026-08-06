"""Orthophoto layer names, by flight year.

Read from the service's GetCapabilities on 2026-08-04. Layer naming is not
regular -- some years carry a resolution suffix and some do not -- so the
mapping is explicit rather than derived. Where a year published more than one
layer, the finest is chosen.

There is no 2022 flight.
"""

#: Flight year to RGB orthophoto layer name.
ORTHO_LAYERS = {
    2014: "Ortoilmakuva_2014",
    2015: "Ortoilmakuva_2015_20cm",
    2016: "Ortoilmakuva_2016",
    2017: "Ortoilmakuva_2017_8cm",
    2018: "Ortoilmakuva_2018",
    2019: "Ortoilmakuva_2019_20cm",
    2020: "Ortoilmakuva_2020",
    2021: "Ortoilmakuva_2021_5cm",
    2023: "Ortoilmakuva_2023_5cm",
    2024: "Ortoilmakuva_2024_5cm",
    2025: "Ortoilmakuva_2025_5cm",
}

LATEST_YEAR = max(ORTHO_LAYERS)


def layer_for_year(year: int) -> str:
    """Layer name for a flight year."""
    try:
        return ORTHO_LAYERS[year]
    except KeyError:
        available = ", ".join(str(y) for y in sorted(ORTHO_LAYERS))
        raise ValueError(
            f"no orthophoto layer for {year}; available: {available}"
        ) from None
