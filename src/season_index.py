"""Central season index mapping."""

from __future__ import annotations

SEASONS: tuple[str, str, str, str] = ("Spring", "Summer", "Autumn", "Winter")
SEASON_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SEASONS)}
IDX_TO_SEASON: dict[int, str] = {i: s for i, s in enumerate(SEASONS)}
