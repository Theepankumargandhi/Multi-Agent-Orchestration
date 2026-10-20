"""One conservative PDF coordinate-frame policy for text and raster probes."""

import math


def page_geometry(page, target):
    media, crop, bounds = page.get_mediabox(fallback_ok=False), page.get_cropbox(), page.get_bbox()
    if media is None or crop is None or bounds is None or page.get_rotation() != 0:
        return False
    expected = (0, 0, target.width, target.height)
    if any(
        not math.isfinite(actual) or abs(actual - wanted) > 0.01
        for actual, wanted in zip(page.get_size(), expected[2:], strict=True)
    ):
        return False
    return all(
        math.isfinite(actual) and abs(actual - wanted) <= 0.01
        for rectangle in (media, crop, bounds)
        for actual, wanted in zip(rectangle, expected, strict=True)
    )
