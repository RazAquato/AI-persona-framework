# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider

import unittest
from unittest.mock import patch, mock_open
from agents.loaders import personality_loader


class TestPersonalityLoader(unittest.TestCase):

    @patch("agents.loaders.personality_loader.open", new_callable=mock_open, read_data="""
emotions:
  joy: {}
  curiosity: {}
  trust: {}
""")
    @patch("yaml.safe_load")
    def test_load_emotion_defaults(self, mock_yaml_load, mock_open_file):
        mock_yaml_load.return_value = {"emotions": {"joy": {}, "curiosity": {}, "trust": {}}}
        result = personality_loader.load_emotion_defaults()
        self.assertEqual(result, {"joy": 0.0, "curiosity": 0.0, "trust": 0.0})

    def test_apply_emotion_defaults_merge(self):
        mock_config = {
            "default_emotions": {"joy": 0.9}
        }

        with patch.object(personality_loader, "load_emotion_defaults") as mock_defaults:
            mock_defaults.return_value = {"joy": 0.0, "curiosity": 0.0}
            updated = personality_loader.apply_emotion_defaults(mock_config)

        self.assertEqual(updated["default_emotions"]["joy"], 0.9)
        self.assertEqual(updated["default_emotions"]["curiosity"], 0.0)

    @patch("agents.loaders.personality_loader.load_user_agents")
    @patch.object(personality_loader, "apply_emotion_defaults")
    def test_load_active_agent_success(self, mock_apply_emotions, mock_load_agents):
        mock_config = {"tone": "gentle"}
        mock_load_agents.return_value = [{"name": "Eva", "config": mock_config}]
        mock_apply_emotions.return_value = mock_config

        result = personality_loader.load_active_agent(user_id=1, agent_name="Eva")
        self.assertEqual(result, mock_config)
        mock_apply_emotions.assert_called_once_with(mock_config)

    @patch("agents.loaders.personality_loader.load_user_agents")
    def test_load_active_agent_not_found(self, mock_load_agents):
        mock_load_agents.return_value = [{"name": "Alice", "config": {}}]
        with self.assertRaises(ValueError) as context:
            personality_loader.load_active_agent(user_id=1, agent_name="Bob")
        self.assertIn("Agent 'Bob' not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()

