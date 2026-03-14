# AI-persona-framework - Persona Store Unit Tests
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import psycopg2
from dotenv import load_dotenv
from memory.persona_store import (
    get_persona, get_persona_by_slug, list_personas,
    create_persona, update_persona, delete_persona,
)


class TestPersonaStore(unittest.TestCase):
    """Integration tests for persona_store CRUD against live PostgreSQL."""

    @classmethod
    def setUpClass(cls):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

        cls.conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        cls.cur = cls.conn.cursor()
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestPersonaUser",))
        cls.user_id = cls.cur.fetchone()[0]
        cls.conn.commit()
        cls.created_ids = []

    @classmethod
    def tearDownClass(cls):
        for pid in cls.created_ids:
            cls.cur.execute("DELETE FROM user_personalities WHERE id = %s;", (pid,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def test_create_persona(self):
        pid = create_persona(
            self.user_id, slug="test_bot", name="TestBot",
            description="A test persona", system_prompt="You are TestBot.",
        )
        self.created_ids.append(pid)
        self.assertIsInstance(pid, int)

    def test_get_persona(self):
        pid = create_persona(self.user_id, slug="get_test", name="GetTest")
        self.created_ids.append(pid)
        persona = get_persona(pid)
        self.assertIsNotNone(persona)
        self.assertEqual(persona["id"], pid)
        self.assertEqual(persona["slug"], "get_test")
        self.assertEqual(persona["name"], "GetTest")
        self.assertEqual(persona["user_id"], self.user_id)

    def test_get_persona_not_found(self):
        result = get_persona(999999)
        self.assertIsNone(result)

    def test_get_persona_by_slug(self):
        pid = create_persona(self.user_id, slug="slug_test", name="SlugTest")
        self.created_ids.append(pid)
        persona = get_persona_by_slug(self.user_id, "slug_test")
        self.assertIsNotNone(persona)
        self.assertEqual(persona["id"], pid)

    def test_get_persona_by_slug_not_found(self):
        result = get_persona_by_slug(self.user_id, "nonexistent_slug_xyz")
        self.assertIsNone(result)

    def test_list_personas(self):
        pid = create_persona(self.user_id, slug="list_test", name="ListTest")
        self.created_ids.append(pid)
        personas = list_personas(self.user_id)
        self.assertIsInstance(personas, list)
        ids = [p["id"] for p in personas]
        self.assertIn(pid, ids)

    def test_list_personas_empty_user(self):
        personas = list_personas(999999)
        self.assertEqual(personas, [])

    def test_update_persona(self):
        pid = create_persona(self.user_id, slug="update_test", name="Before")
        self.created_ids.append(pid)
        result = update_persona(pid, name="After", description="Updated")
        self.assertTrue(result)
        persona = get_persona(pid)
        self.assertEqual(persona["name"], "After")
        self.assertEqual(persona["description"], "Updated")

    def test_update_persona_not_found(self):
        result = update_persona(999999, name="Nope")
        self.assertFalse(result)

    def test_delete_persona(self):
        pid = create_persona(self.user_id, slug="delete_test", name="ToDelete")
        result = delete_persona(pid)
        self.assertTrue(result)
        self.assertIsNone(get_persona(pid))

    def test_delete_persona_not_found(self):
        result = delete_persona(999999)
        self.assertFalse(result)

    def test_persona_includes_nsfw_fields(self):
        pid = create_persona(
            self.user_id, slug="nsfw_test", name="NsfwTest",
            nsfw_capable=True, nsfw_prompt_addon="NSFW addon.",
        )
        self.created_ids.append(pid)
        persona = get_persona(pid)
        self.assertTrue(persona["nsfw_capable"])
        self.assertEqual(persona["nsfw_prompt_addon"], "NSFW addon.")
        # Alias for prompt_builder
        self.assertEqual(persona["nsfw_system_prompt_addon"], "NSFW addon.")

    def test_persona_default_memory_scope(self):
        pid = create_persona(self.user_id, slug="scope_test", name="ScopeTest")
        self.created_ids.append(pid)
        persona = get_persona(pid)
        self.assertIn("memory_scope", persona)
        self.assertTrue(persona["memory_scope"]["tier1"])

    def test_unique_slug_per_user(self):
        pid = create_persona(self.user_id, slug="unique_slug", name="First")
        self.created_ids.append(pid)
        with self.assertRaises(Exception):
            create_persona(self.user_id, slug="unique_slug", name="Duplicate")

    # --- Phase 1: Domain access ---

    def test_create_persona_with_domain_access(self):
        pid = create_persona(self.user_id, slug="domain_test", name="DomainTest",
                             domain_access=["work", "hobbies"])
        self.created_ids.append(pid)
        persona = get_persona(pid)
        self.assertEqual(set(persona["domain_access"]), {"work", "hobbies"})

    def test_update_persona_domain_access(self):
        pid = create_persona(self.user_id, slug="domain_upd", name="DomainUpd",
                             domain_access=["work"])
        self.created_ids.append(pid)
        update_persona(pid, domain_access=["family", "physical", "work"])
        persona = get_persona(pid)
        self.assertEqual(set(persona["domain_access"]), {"family", "physical", "work"})

    def test_persona_domain_access_defaults_empty(self):
        pid = create_persona(self.user_id, slug="domain_empty", name="DomainEmpty")
        self.created_ids.append(pid)
        persona = get_persona(pid)
        self.assertEqual(persona["domain_access"], [])

    def test_list_personas_includes_domain_access(self):
        pid = create_persona(self.user_id, slug="domain_list", name="DomainList",
                             domain_access=["emotional", "memories"])
        self.created_ids.append(pid)
        personas = list_personas(self.user_id)
        found = [p for p in personas if p["id"] == pid]
        self.assertEqual(len(found), 1)
        self.assertIn("domain_access", found[0])
        self.assertEqual(set(found[0]["domain_access"]), {"emotional", "memories"})


if __name__ == "__main__":
    unittest.main()
