# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import tempfile
from unittest.mock import patch

from tools.document_ingest import ingest_document, _read_file, _split_into_chunks


class TestDocumentIngest(unittest.TestCase):

    def test_file_not_found(self):
        result = ingest_document("/nonexistent/file.txt", user_id=9999)
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"data")
            f.flush()
            result = ingest_document(f.name, user_id=9999)
        os.unlink(f.name)
        self.assertFalse(result["success"])
        self.assertIn("Unsupported", result["error"])

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            result = ingest_document(f.name, user_id=9999)
        os.unlink(f.name)
        self.assertFalse(result["success"])
        self.assertIn("empty", result["error"])

    @patch("tools.document_ingest.ingest_facts", return_value={"facts_stored": 2, "facts_skipped": 0, "topics_found": 1, "topic_names": ["hiking"]})
    def test_ingest_txt_file(self, mock_ingest):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("My name is Kenneth and I live in Norway.\n\nI love hiking in the mountains.")
            f.flush()
            result = ingest_document(f.name, user_id=9999)
        os.unlink(f.name)
        self.assertTrue(result["success"])
        self.assertGreater(result["total_extracted"], 0)
        mock_ingest.assert_called_once()

    @patch("tools.document_ingest.ingest_facts", return_value={"facts_stored": 0, "facts_skipped": 0, "topics_found": 0, "topic_names": []})
    def test_ingest_md_file(self, mock_ingest):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Notes\n\nJust some random text about nothing in particular.")
            f.flush()
            result = ingest_document(f.name, user_id=9999)
        os.unlink(f.name)
        self.assertTrue(result["success"])

    @patch("tools.document_ingest.ingest_facts", return_value={"facts_stored": 1, "facts_skipped": 0, "topics_found": 0, "topic_names": []})
    def test_ingest_json_file(self, mock_ingest):
        import json
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"note": "I work as a software engineer"}, f)
            f.flush()
            result = ingest_document(f.name, user_id=9999)
        os.unlink(f.name)
        self.assertTrue(result["success"])


class TestReadFile(unittest.TestCase):

    def test_read_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("hello world")
            f.flush()
            result = _read_file(f.name, ".txt")
        os.unlink(f.name)
        self.assertEqual(result, "hello world")

    def test_read_json_dict(self):
        import json
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            result = _read_file(f.name, ".json")
        os.unlink(f.name)
        self.assertIn("key", result)


class TestSplitChunks(unittest.TestCase):

    def test_split_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        chunks = _split_into_chunks(text, ".txt")
        self.assertEqual(len(chunks), 3)

    def test_split_single_block(self):
        text = "Just one block of text."
        chunks = _split_into_chunks(text, ".txt")
        self.assertEqual(len(chunks), 1)

    def test_split_json_lines(self):
        text = '{"a": 1}\n{"b": 2}\n{"c": 3}'
        chunks = _split_into_chunks(text, ".jsonl")
        self.assertEqual(len(chunks), 3)


if __name__ == "__main__":
    unittest.main()
