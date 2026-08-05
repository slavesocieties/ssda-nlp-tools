"""volume_geo.py — the institution and place that produced each record.

Daniel, 2026-08-05: "there's also foundational location data based on the
institution that produced the record ... This is absolutely critical to
disambiguation and should be considered heavily in any computational
deliberation."

He is right and we were not using it. The disambiguator knew only `_register`
(the volume id), which is an opaque token: it can tell "same volume" from
"different volume" and nothing else. It could not tell that 201991 (Guanabacoa)
and 29597 (Santo Angel Custodio) are two Havana parishes ten kilometres apart,
while 701179 is in Rio de Janeiro state.

WHAT volumes.json ACTUALLY CARRIES, per volume, under `fields`:

    institution   "Iglesia de Nuestra Senora de la Asuncion de Guanabacoa"
    city/state/country            Havana / La Habana / Cuba
    coords        "23.09992, 82.31738"
    start_date / end_date         the volume's covered period
    language

THE COORDINATES ARE NOT TRUSTWORTHY AND MUST BE REPAIRED FIRST.
--------------------------------------------------------------
Every institution in this collection is in the Americas or West Africa, so every
longitude must be negative. **42 of 397 Cuban volumes (10.6%) carry a POSITIVE
longitude**, i.e. the minus sign is missing; Brazil, Colombia, Mexico and the
United States are clean. 201991 is one of them, and it is our largest volume at
2,019 records.

Used raw, that single missing character puts Guanabacoa in the Indian Ocean:
201991 reads as 14,613 km from Santo Angel Custodio, when the two parishes are
about ten kilometres apart. A distance feature built on the raw field would have
made our biggest volume maximally distant from everything and looked like
evidence rather than a typo.

So: longitudes are sign-corrected on load, the repair is COUNTED and reportable
rather than silent, and `same_place` prefers the structured city/state/country
strings -- which have no sign to get wrong -- over the coordinates.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

# Every SSDA institution sits west of Greenwich.
_MAX_VALID_LON = 0.0


class VolumeGeo:
    """Institution and place per volume id, with coordinates repaired."""

    def __init__(self, path: str):
        self.by_id: Dict[str, dict] = {}
        self.repaired: list = []
        self.missing_coords: list = []
        raw = json.load(open(path, encoding="utf-8"))
        vols = raw.get("volumes") if isinstance(raw, dict) else raw
        for v in vols or []:
            vid = str(v.get("id") or "")
            f = v.get("fields") or {}
            if not vid:
                continue
            lat, lon = self._coords(f.get("coords"))
            if lon is not None and lon > _MAX_VALID_LON:
                self.repaired.append((vid, lon))
                lon = -lon
            if lat is None:
                self.missing_coords.append(vid)
            self.by_id[vid] = {
                "institution": f.get("institution"),
                "city": f.get("city"), "state": f.get("state"),
                "country": self._one(f.get("country")),
                "language": self._one(f.get("language")),
                "lat": lat, "lon": lon,
                "start": f.get("start_date"), "end": f.get("end_date"),
            }

    @staticmethod
    def _one(v):
        return (v[0] if isinstance(v, list) and v else v)

    @staticmethod
    def _coords(s) -> Tuple[Optional[float], Optional[float]]:
        try:
            lat, lon = [float(x) for x in str(s).split(",")]
            return lat, lon
        except Exception:
            return None, None

    def get(self, volume_id) -> Optional[dict]:
        return self.by_id.get(str(volume_id))

    def km_between(self, a, b) -> Optional[float]:
        ga, gb = self.get(a), self.get(b)
        if not ga or not gb:
            return None
        if None in (ga["lat"], ga["lon"], gb["lat"], gb["lon"]):
            return None
        R = 6371.0
        p1, p2 = math.radians(ga["lat"]), math.radians(gb["lat"])
        dp = math.radians(gb["lat"] - ga["lat"])
        dl = math.radians(gb["lon"] - ga["lon"])
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(h))

    def same_institution(self, a, b) -> Optional[bool]:
        ga, gb = self.get(a), self.get(b)
        if not ga or not gb or not ga["institution"] or not gb["institution"]:
            return None
        return ga["institution"].strip().lower() == gb["institution"].strip().lower()

    def same_place(self, a, b) -> Optional[str]:
        """Coarsest shared administrative level, preferring the STRUCTURED
        fields: they cannot carry the sign defect that corrupts the coordinates.

        Returns "institution" | "city" | "state" | "country" | "none", or None
        when either volume is unknown.
        """
        ga, gb = self.get(a), self.get(b)
        if not ga or not gb:
            return None
        if self.same_institution(a, b):
            return "institution"
        for level in ("city", "state", "country"):
            va, vb = ga.get(level), gb.get(level)
            if va and vb and str(va).strip().lower() == str(vb).strip().lower():
                return level
        return "none"

    def overlapping_years(self, a, b) -> Optional[bool]:
        """Do the two volumes even cover a common period? Two registers whose
        date ranges never meet cannot hold the same living person twice."""
        ga, gb = self.get(a), self.get(b)
        if not ga or not gb:
            return None
        def yr(s):
            try:
                return int(str(s)[:4])
            except Exception:
                return None
        a0, a1, b0, b1 = yr(ga["start"]), yr(ga["end"]), yr(gb["start"]), yr(gb["end"])
        if None in (a0, a1, b0, b1):
            return None
        return not (a1 < b0 or b1 < a0)

    def report(self) -> str:
        lines = [f"{len(self.by_id):,} volumes loaded"]
        if self.repaired:
            lines.append(f"  {len(self.repaired):,} longitude sign(s) REPAIRED "
                         f"(positive longitude is impossible in this collection)")
            for vid, lon in self.repaired[:5]:
                lines.append(f"    {vid}: {lon} -> {-lon}")
        if self.missing_coords:
            lines.append(f"  {len(self.missing_coords):,} volumes have no coordinates")
        return "\n".join(lines)


_CACHE: Dict[str, VolumeGeo] = {}


def load(path: str = "../ssda-openai/volumes.json") -> Optional[VolumeGeo]:
    if not os.path.exists(path):
        return None
    if path not in _CACHE:
        _CACHE[path] = VolumeGeo(path)
    return _CACHE[path]
