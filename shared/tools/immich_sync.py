# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Immich Sync Adapter
-------------------
Fetches photo archive metadata from an Immich instance and converts it into
identity-tier facts for the user's knowledge profile.

Extracts three categories of facts from the photo library:

  People facts:
    Named people recognized by face detection, ranked by photo frequency.
    "User has many photos with Ylva (707 photos)" → high confidence
    "User has photos with Birgitte (3 photos)" → lower confidence

  Location facts:
    Cities and countries where photos were taken, based on EXIF GPS data.
    Sampled across the full timeline to avoid recency bias.
    "User has taken many photos in Lysaker" / "User has visited France"

  Device facts:
    Camera devices used, indicating what gear the user owns/has owned.
    "User takes photos with Apple iPhone 13 mini"

Each sync is a full snapshot: old immich facts are deleted and replaced
with fresh ones. Designed to run as a monthly cron job.

All facts use source_type="immich" for clean snapshot replacement.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "memory-server"))
for p in [SHARED_PATH, MEMORY_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from memory.fact_store import store_fact_blobs, make_fact_blob, delete_facts_by_source


class ImmichClient:
    """Minimal HTTP client for the Immich REST API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        req.add_header("x-api-key", self.api_key)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("x-api-key", self.api_key)
        req.add_header("Accept", "application/json")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def get_people(self) -> list:
        """Fetch all named, visible people."""
        data = self._get("/api/people")
        return [p for p in data.get("people", []) if p.get("name") and not p.get("isHidden")]

    def count_person_photos(self, person_id: str) -> int:
        """Count photos containing a specific person by paginating search."""
        total = 0
        page = 1
        while True:
            result = self._post("/api/search/metadata", {
                "personIds": [person_id], "size": 250, "page": page,
            })
            assets = result.get("assets", {})
            total += assets.get("count", 0)
            if not assets.get("nextPage"):
                break
            page += 1
            if page > 50:
                break
        return total

    def get_asset_detail(self, asset_id: str) -> dict:
        """Fetch full asset detail including EXIF and people."""
        return self._get(f"/api/assets/{asset_id}")

    def search_assets(self, page: int = 1, size: int = 250, **filters) -> dict:
        """Search assets by metadata filters. Returns raw response."""
        body = {"page": page, "size": size, **filters}
        return self._post("/api/search/metadata", body)

    def sample_asset_details(self, year: str, sample_size: int = 15) -> list:
        """Fetch a sample of asset details from a given year for EXIF data."""
        result = self.search_assets(
            page=1, size=sample_size, type="IMAGE",
            takenAfter=f"{year}-01-01T00:00:00.000Z",
            takenBefore=f"{year}-12-31T23:59:59.999Z",
        )
        details = []
        for item in result.get("assets", {}).get("items", []):
            try:
                detail = self.get_asset_detail(item["id"])
                details.append(detail)
            except (urllib.error.URLError, Exception):
                pass
        return details


def sync_immich(user_id: int, base_url: str = None, api_key: str = None,
                dry_run: bool = False) -> dict:
    """
    Sync photo archive metadata from Immich into the fact store.

    Extracts people (by face frequency), locations (by EXIF GPS), and
    devices (by camera model). Old immich facts are replaced with a
    fresh snapshot.

    Args:
        user_id: the user who owns these facts
        base_url: Immich server URL (falls back to IMMICH_URL env var)
        api_key: Immich API key (falls back to IMMICH_API_KEY env var)
        dry_run: if True, return facts without storing them

    Returns:
        dict with keys: people_found, locations_found, devices_found,
                        facts_generated, facts, deleted_old
    """
    base_url = base_url or os.getenv("IMMICH_URL")
    api_key = api_key or os.getenv("IMMICH_API_KEY")

    if not base_url or not api_key:
        return {"error": "IMMICH_URL and IMMICH_API_KEY must be set",
                "people_found": 0, "locations_found": 0, "devices_found": 0,
                "facts_generated": 0, "facts": [], "deleted_old": 0}

    client = ImmichClient(base_url, api_key)

    # 1. People — named faces with photo frequency
    people = client.get_people()
    person_counts = {}
    for p in people:
        try:
            count = client.count_person_photos(p["id"])
            if count > 0:
                person_counts[p["name"]] = count
        except (urllib.error.URLError, Exception):
            pass

    # 2. Locations & devices — sample EXIF data across years
    city_counter = Counter()
    country_counter = Counter()
    device_counter = Counter()

    # Sample across a range of years to get a broad picture
    for year in range(2018, 2027):
        try:
            details = client.sample_asset_details(str(year), sample_size=20)
            for detail in details:
                exif = detail.get("exifInfo", {})
                if exif.get("city"):
                    city_counter[exif["city"]] += 1
                if exif.get("country"):
                    country_counter[exif["country"]] += 1
                make = exif.get("make", "")
                model = exif.get("model", "")
                if make and model:
                    device_counter[f"{make} {model}"] += 1
        except (urllib.error.URLError, Exception):
            pass

    # 3. Build facts
    facts = []

    # People facts — confidence scales with photo count
    for name, count in sorted(person_counts.items(), key=lambda x: -x[1]):
        if count >= 50:
            text = f"User has many photos with {name} ({count} photos)"
            confidence = 0.9
        elif count >= 10:
            text = f"User has photos with {name} ({count} photos)"
            confidence = 0.75
        else:
            text = f"User has a few photos with {name}"
            confidence = 0.55

        facts.append(make_fact_blob(
            text=text,
            tier="identity",
            tags=["immich", "person"],
            source_type="immich",
            source_ref=f"person:{name}",
            confidence=confidence,
            entity_type="person",
        ))

    # City facts — only cities appearing in 2+ sampled photos
    for city, count in city_counter.most_common():
        if count >= 3:
            text = f"User has taken many photos in {city}"
            confidence = 0.8
        elif count >= 2:
            text = f"User has taken photos in {city}"
            confidence = 0.65
        else:
            continue

        facts.append(make_fact_blob(
            text=text,
            tier="identity",
            tags=["immich", "location"],
            source_type="immich",
            source_ref=f"city:{city}",
            confidence=confidence,
        ))

    # Country facts — countries other than primary (Norway assumed)
    for country, count in country_counter.most_common():
        if count >= 2:
            facts.append(make_fact_blob(
                text=f"User has visited {country}",
                tier="identity",
                tags=["immich", "location", "country"],
                source_type="immich",
                source_ref=f"country:{country}",
                confidence=0.7,
            ))

    # Device facts — cameras appearing in 3+ sampled photos
    for device, count in device_counter.most_common():
        if count >= 3:
            facts.append(make_fact_blob(
                text=f"User takes photos with {device}",
                tier="identity",
                tags=["immich", "device"],
                source_type="immich",
                source_ref=f"device:{device}",
                confidence=0.7,
            ))

    # 4. Store: delete old snapshot, write new one
    deleted_old = 0
    if not dry_run:
        deleted_old = delete_facts_by_source(user_id, "immich")
        store_fact_blobs(user_id, facts, source_type="immich")

    return {
        "people_found": len(person_counts),
        "locations_found": len(city_counter),
        "devices_found": len(device_counter),
        "facts_generated": len(facts),
        "facts": facts,
        "deleted_old": deleted_old,
    }
