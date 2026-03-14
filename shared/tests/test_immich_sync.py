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

import unittest
from unittest.mock import patch, MagicMock
from tools.immich_sync import ImmichClient, sync_immich


FAKE_PEOPLE = {
    "people": [
        {"id": "p1", "name": "Ylva", "isHidden": False},
        {"id": "p2", "name": "Trym", "isHidden": False},
        {"id": "p3", "name": "Kenneth", "isHidden": False},
        {"id": "p4", "name": "Birgitte", "isHidden": False},
        {"id": "p5", "name": "", "isHidden": False},       # unnamed → filtered
        {"id": "p6", "name": "Hidden", "isHidden": True},   # hidden → filtered
    ],
    "total": 6,
    "hidden": 1,
}

PERSON_PHOTO_COUNTS = {
    "p1": 707,   # Ylva — many photos
    "p2": 641,   # Trym — many photos
    "p3": 103,   # Kenneth — medium
    "p4": 3,     # Birgitte — few
}


def _fake_search_metadata(body):
    """Simulate /api/search/metadata for person photo counts."""
    person_ids = body.get("personIds", [])
    if person_ids:
        pid = person_ids[0]
        count = PERSON_PHOTO_COUNTS.get(pid, 0)
        return {"assets": {"count": count, "items": [], "nextPage": None}}

    # Year-based search for EXIF sampling
    year = body.get("takenAfter", "")[:4]
    if year:
        return {"assets": {
            "count": 3,
            "items": [
                {"id": f"asset-{year}-1"},
                {"id": f"asset-{year}-2"},
                {"id": f"asset-{year}-3"},
            ],
            "nextPage": None,
        }}
    return {"assets": {"count": 0, "items": [], "nextPage": None}}


FAKE_ASSET_DETAILS = {
    "asset-2024-1": {
        "exifInfo": {"city": "Lysaker", "country": "Norway",
                     "make": "Apple", "model": "iPhone 13 mini"},
    },
    "asset-2024-2": {
        "exifInfo": {"city": "Lysaker", "country": "Norway",
                     "make": "Apple", "model": "iPhone 13 mini"},
    },
    "asset-2024-3": {
        "exifInfo": {"city": "Sandvika", "country": "Norway",
                     "make": "Sony", "model": "J9110"},
    },
    "asset-2025-1": {
        "exifInfo": {"city": "Lysaker", "country": "Norway",
                     "make": "Apple", "model": "iPhone 13 mini"},
    },
    "asset-2025-2": {
        "exifInfo": {"city": "Paris", "country": "France",
                     "make": "Apple", "model": "iPhone 13 mini"},
    },
    "asset-2025-3": {
        "exifInfo": {"city": "Paris", "country": "France",
                     "make": "Apple", "model": "iPhone 13 mini"},
    },
}


def _mock_get(path):
    if path == "/api/people":
        return FAKE_PEOPLE
    # Asset detail
    for asset_id, detail in FAKE_ASSET_DETAILS.items():
        if path.endswith(f"/{asset_id}"):
            return detail
    return {}


def _mock_post(path, body):
    if "/search/metadata" in path:
        return _fake_search_metadata(body)
    return {}


class TestImmichClient(unittest.TestCase):

    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_get_people_filters_unnamed_and_hidden(self, mock_get):
        client = ImmichClient("http://fake", "key")
        people = client.get_people()
        names = [p["name"] for p in people]
        self.assertIn("Ylva", names)
        self.assertNotIn("", names)
        self.assertNotIn("Hidden", names)
        self.assertEqual(len(people), 4)

    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    def test_count_person_photos(self, mock_post):
        client = ImmichClient("http://fake", "key")
        count = client.count_person_photos("p1")
        self.assertEqual(count, 707)


class TestSyncImmich(unittest.TestCase):

    @patch.dict("os.environ", {"IMMICH_URL": "", "IMMICH_API_KEY": ""})
    def test_missing_credentials(self):
        result = sync_immich(user_id=9999, base_url=None, api_key=None)
        self.assertIn("error", result)

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_sync_produces_people_facts(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        self.assertEqual(result["people_found"], 4)

        people_facts = [f for f in result["facts"] if "person" in f["tags"]]
        self.assertEqual(len(people_facts), 4)

        ylva_fact = next(f for f in people_facts if "Ylva" in f["text"])
        self.assertIn("many photos", ylva_fact["text"])
        self.assertIn("707", ylva_fact["text"])

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_confidence_scales_with_photo_count(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        people_facts = [f for f in result["facts"] if "person" in f["tags"]]

        ylva = next(f for f in people_facts if "Ylva" in f["text"])
        birgitte = next(f for f in people_facts if "Birgitte" in f["text"])
        self.assertGreater(ylva["confidence"], birgitte["confidence"])

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_sync_produces_location_facts(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        location_facts = [f["text"] for f in result["facts"] if "location" in f["tags"]]
        self.assertTrue(any("Lysaker" in f for f in location_facts))

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_sync_produces_country_facts(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        country_facts = [f["text"] for f in result["facts"] if "country" in f["tags"]]
        self.assertTrue(any("Norway" in f for f in country_facts))
        self.assertTrue(any("France" in f for f in country_facts))

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_sync_produces_device_facts(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        device_facts = [f["text"] for f in result["facts"] if "device" in f["tags"]]
        self.assertTrue(any("iPhone 13 mini" in f for f in device_facts))

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_all_facts_are_identity_tier(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        for fact in result["facts"]:
            self.assertEqual(fact["tier"], "identity")

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_all_facts_have_immich_source_type(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        for fact in result["facts"]:
            self.assertEqual(fact["source_type"], "immich")

    @patch("tools.immich_sync.delete_facts_by_source", return_value=5)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_sync_deletes_old_facts_before_insert(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        mock_delete.assert_called_once_with(9999, "immich")
        mock_store.assert_called_once()
        self.assertEqual(result["deleted_old"], 5)

    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_dry_run_does_not_store(self, mock_get, mock_post):
        with patch("tools.immich_sync.store_fact_blobs") as mock_store, \
             patch("tools.immich_sync.delete_facts_by_source") as mock_delete:
            result = sync_immich(user_id=9999, base_url="http://fake",
                                 api_key="key", dry_run=True)
            mock_store.assert_not_called()
            mock_delete.assert_not_called()
            self.assertGreater(result["facts_generated"], 0)

    @patch.object(ImmichClient, '_post', return_value={"assets": {"count": 0, "items": [], "nextPage": None}})
    @patch.object(ImmichClient, '_get', return_value={"people": [], "total": 0, "hidden": 0})
    def test_empty_library_returns_zero(self, mock_get, mock_post):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key",
                             dry_run=True)
        self.assertEqual(result["people_found"], 0)
        self.assertEqual(result["facts_generated"], 0)

    @patch("tools.immich_sync.delete_facts_by_source", return_value=0)
    @patch("tools.immich_sync.store_fact_blobs")
    @patch.object(ImmichClient, '_post', side_effect=_mock_post)
    @patch.object(ImmichClient, '_get', side_effect=_mock_get)
    def test_people_facts_have_entity_type_person(self, mock_get, mock_post, mock_store, mock_delete):
        result = sync_immich(user_id=9999, base_url="http://fake", api_key="key")
        people_facts = [f for f in result["facts"] if "person" in f["tags"]]
        for fact in people_facts:
            self.assertEqual(fact["entity_type"], "person")


if __name__ == "__main__":
    unittest.main()
