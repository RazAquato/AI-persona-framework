import os
import unittest
import psycopg2
from dotenv import load_dotenv

class TestPostgresInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
        load_dotenv(dotenv_path=env_path)

        cls.conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        cls.cur = cls.conn.cursor()

    def test_expected_tables_exist(self):
        expected_tables = {
            'users',
            'user_personalities',
            'chat_sessions',
            'chat_messages',
            'facts',
            'topic_tags',
            'message_metadata',
            'emotional_relationships'
        }

        self.cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        actual_tables = {row[0] for row in self.cur.fetchall()}

        missing = expected_tables - actual_tables
        self.assertFalse(missing, f"Missing expected tables: {missing}")

    def test_no_unexpected_tables_exist(self):
        expected_tables = {
            'users',
            'user_personalities',
            'chat_sessions',
            'chat_messages',
            'facts',
            'topic_tags',
            'message_metadata',
            'emotional_relationships'
        }

        self.cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        actual_tables = {row[0] for row in self.cur.fetchall()}

        unexpected = actual_tables - expected_tables
        self.assertFalse(unexpected, f"Unexpected tables found: {unexpected}")

    def test_vector_extension_installed(self):
        self.cur.execute("SELECT extname FROM pg_extension;")
        extensions = {row[0] for row in self.cur.fetchall()}
        self.assertIn("vector", extensions)

    @classmethod
    def tearDownClass(cls):
        cls.cur.close()
        cls.conn.close()

