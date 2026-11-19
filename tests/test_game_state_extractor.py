import unittest
from unittest.mock import Mock, patch

import requests

from perception.game_state_extractor import (
    GAME_STATE_SCHEMA,
    build_prompt,
    extract_game_state,
    validate_game_state,
)


def valid_state():
    return {
        "screen_type": "gameplay",
        "player": {"name": None, "team_or_color": None},
        "game_status": {"mode": None, "timer": None},
        "map": {"visible": True, "terrain": [], "regions": []},
        "entities": {
            "cities": [{
                "type": "city",
                "approximate_position": {"x": 0.25, "y": 0.75},
                "visible_owner_or_color": None,
                "confidence": 0.8,
            }],
            "castles": [],
            "units": [],
            "other_structures": [],
        },
        "interface": {"resources": [], "visible_buttons": [], "visible_text": []},
        "spatial_observations": [],
        "uncertainties": ["Small labels are unreadable."],
    }


class GameStateExtractorTests(unittest.TestCase):
    def test_prompt_requires_evidence_and_prohibits_filler(self):
        prompt = build_prompt().lower()
        self.assertIn("direct visual evidence", prompt)
        self.assertIn("do not output alphabet sequences", prompt)
        self.assertIn("normalized x/y coordinates", prompt)

    def test_schema_requires_all_top_level_sections(self):
        self.assertEqual(
            set(GAME_STATE_SCHEMA["required"]),
            {
                "screen_type", "player", "game_status", "map", "entities",
                "interface", "spatial_observations", "uncertainties",
            },
        )

    def test_valid_game_state_is_accepted(self):
        result = valid_state()
        self.assertIs(validate_game_state(result), result)

    def test_missing_top_level_field_is_rejected(self):
        result = valid_state()
        del result["uncertainties"]
        with self.assertRaisesRegex(ValueError, "Invalid game-state fields"):
            validate_game_state(result)

    def test_invalid_screen_type_is_rejected(self):
        result = valid_state()
        result["screen_type"] = "alphabet soup"
        with self.assertRaisesRegex(ValueError, "screen_type"):
            validate_game_state(result)

    def test_out_of_range_entity_coordinate_is_rejected(self):
        result = valid_state()
        result["entities"]["cities"][0]["approximate_position"]["x"] = 1.4
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            validate_game_state(result)

    def test_non_numeric_confidence_is_rejected(self):
        result = valid_state()
        result["entities"]["cities"][0]["confidence"] = "high"
        with self.assertRaisesRegex(ValueError, "confidence"):
            validate_game_state(result)

    def test_truncated_ollama_response_is_rejected(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"done_reason": "length"}
        with patch("perception.game_state_extractor.encode_image", return_value="encoded"):
            with patch("perception.game_state_extractor.requests.post", return_value=response):
                with self.assertRaisesRegex(ValueError, "truncated"):
                    extract_game_state("unused.png")

    def test_extractor_uses_json_schema_and_validates_model_response(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "done_reason": "stop",
            "message": {"content": __import__("json").dumps(valid_state())},
        }
        with patch("perception.game_state_extractor.encode_image", return_value="encoded"):
            with patch("perception.game_state_extractor.requests.post", return_value=response) as post:
                result = extract_game_state("unused.png")
        self.assertEqual(result, valid_state())
        self.assertEqual(post.call_args.kwargs["json"]["format"], GAME_STATE_SCHEMA)
        self.assertEqual(post.call_args.kwargs["json"]["options"]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
