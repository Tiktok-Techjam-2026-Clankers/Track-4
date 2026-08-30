"""Unit tests for the LLM intent parser layer.

All OpenAI API interactions are mocked — no real network calls.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from starter.intent_parser import (
    DeterministicIntentParser,
    OpenAIIntentParser,
    HybridIntentParser,
    IntentResult,
    LLM_CONFIDENCE_THRESHOLD,
    load_api_key,
    validate_intent_result,
)
from starter.agent import IntentClassifier


def _openai_response(payload: dict, prompt_tokens: int = 10, completion_tokens: int = 5) -> bytes:
    """Build a fake OpenAI chat completions JSON response."""
    return json.dumps({
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }).encode()


def _mock_urlopen(data: bytes, status: int = 200):
    """Return a context-manager mock that yields *data* from read()."""
    response = MagicMock()
    response.read.return_value = data
    response.status = status
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    return response


class ValidateIntentResultTest(unittest.TestCase):
    def test_valid_full_payload(self) -> None:
        result = validate_intent_result({
            "mode": "override",
            "operation": "replace",
            "category": "shoes",
            "add_constraints": {"color": "white"},
            "remove_constraints": ["black"],
            "negative_constraints": ["leather"],
            "no_preference": ["brand"],
            "referenced_previous_item": True,
            "reference_description": "the first pair",
            "confidence": 0.9,
        })
        assert result is not None
        self.assertEqual(result.mode, "override")
        self.assertEqual(result.operation, "replace")
        self.assertEqual(result.category, "shoes")
        self.assertEqual(result.add_constraints, {"color": "white"})
        self.assertEqual(result.remove_constraints, ["black"])
        self.assertEqual(result.negative_constraints, ["leather"])
        self.assertEqual(result.no_preference, ["brand"])
        self.assertTrue(result.referenced_previous_item)
        self.assertEqual(result.reference_description, "the first pair")
        self.assertAlmostEqual(result.confidence, 0.9)
        self.assertEqual(result.source, "llm")

    def test_minimal_valid_payload(self) -> None:
        result = validate_intent_result({
            "mode": "browsing",
            "operation": "none",
            "confidence": 0.5,
        })
        assert result is not None
        self.assertEqual(result.mode, "browsing")
        self.assertEqual(result.add_constraints, {})
        self.assertEqual(result.negative_constraints, [])

    def test_rejects_invalid_mode(self) -> None:
        self.assertIsNone(validate_intent_result({"mode": "shopping", "operation": "add", "confidence": 1}))

    def test_rejects_invalid_operation(self) -> None:
        self.assertIsNone(validate_intent_result({"mode": "browsing", "operation": "destroy", "confidence": 1}))

    def test_rejects_non_dict(self) -> None:
        self.assertIsNone(validate_intent_result("not a dict"))
        self.assertIsNone(validate_intent_result(None))
        self.assertIsNone(validate_intent_result([]))

    def test_clamps_confidence(self) -> None:
        result = validate_intent_result({"mode": "buying", "operation": "add", "confidence": 5.0})
        assert result is not None
        self.assertEqual(result.confidence, 1.0)

        result = validate_intent_result({"mode": "buying", "operation": "add", "confidence": -2.0})
        assert result is not None
        self.assertEqual(result.confidence, 0.0)

    def test_handles_bad_confidence_type(self) -> None:
        result = validate_intent_result({"mode": "buying", "operation": "add", "confidence": "high"})
        assert result is not None
        self.assertEqual(result.confidence, 0.0)

    def test_coerces_list_entries(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.8,
            "negative_constraints": [123, None, "leather", ""],
        })
        assert result is not None
        self.assertEqual(result.negative_constraints, ["123", "leather"])

    def test_product_ids_in_constraints_are_inert(self) -> None:
        """add_constraints pass validation but the agent never uses them as IDs."""
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "add_constraints": {"parent_asin": "B08XYZ123"},
        })
        assert result is not None
        self.assertIn("parent_asin", result.add_constraints)
        self.assertNotIn("B08XYZ123", [result.category or ""])


class DeterministicIntentParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = IntentClassifier()
        self.parser = DeterministicIntentParser(self.classifier)

    def test_fallback_returns_browsing(self) -> None:
        result = self.parser.parse("just looking at shoes", {"active_query": ""})
        self.assertEqual(result.mode, "browsing")
        self.assertEqual(result.source, "deterministic")
        self.assertEqual(result.confidence, 1.0)

    def test_all_messages_return_browsing(self) -> None:
        result = self.parser.parse("black leather running shoes under $80", {"active_query": ""})
        self.assertEqual(result.mode, "browsing")

    def test_override_message_returns_browsing(self) -> None:
        result = self.parser.parse("Actually, make them white instead", {"active_query": ""})
        self.assertEqual(result.mode, "browsing")

    def test_no_token_usage(self) -> None:
        result = self.parser.parse("shoes", {"active_query": ""})
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)


class OpenAIIntentParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = OpenAIIntentParser(api_key="test-key", timeout=2.0)

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_successful_parse(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "mode": "override",
            "operation": "replace",
            "confidence": 0.9,
            "add_constraints": {"color": "white"},
            "negative_constraints": [],
            "no_preference": [],
        }, prompt_tokens=50, completion_tokens=20))

        result = self.parser.parse("On second thought, white would suit me better.", {
            "active_query": "black shoes", "turn": 3, "session_mode": "buying",
            "last_question": "color",
        })
        self.assertEqual(result.mode, "override")
        self.assertEqual(result.source, "llm")
        self.assertAlmostEqual(result.confidence, 0.9)
        self.assertEqual(result.prompt_tokens, 50)
        self.assertEqual(result.completion_tokens, 20)

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_timeout_raises(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = TimeoutError("timed out")
        with self.assertRaises(RuntimeError):
            self.parser.parse("hello", {"active_query": "", "turn": 1})

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_malformed_json_raises(self, mock_urlopen) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "not json at all"}}],
            "usage": {},
        }).encode()
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = response

        with self.assertRaises(RuntimeError):
            self.parser.parse("hello", {"active_query": "", "turn": 1})

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_invalid_schema_raises(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "mode": "INVALID_MODE",
            "operation": "add",
            "confidence": 0.5,
        }))
        with self.assertRaises(ValueError):
            self.parser.parse("hello", {"active_query": "", "turn": 1})

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_caches_identical_requests(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "mode": "buying", "operation": "add", "confidence": 0.8,
        }))
        state = {"active_query": "shoes", "turn": 1, "session_mode": "browsing", "last_question": None}
        result1 = self.parser.parse("cotton shoes", state)
        result2 = self.parser.parse("cotton shoes", state)
        self.assertEqual(mock_urlopen.call_count, 1)
        self.assertEqual(result1.mode, result2.mode)

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_empty_choices_raises(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(json.dumps({
            "choices": [],
            "usage": {"prompt_tokens": 10},
        }).encode())
        with self.assertRaises(RuntimeError):
            self.parser.parse("hello", {"active_query": "", "turn": 1})

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_network_error_raises(self, mock_urlopen) -> None:
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with self.assertRaises(RuntimeError):
            self.parser.parse("hello", {"active_query": "", "turn": 1})

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_never_leaks_api_key_in_prompt(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "mode": "browsing", "operation": "none", "confidence": 0.5,
        }))
        self.parser.parse("hello", {"active_query": "", "turn": 1})
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data)
        for msg in body["messages"]:
            self.assertNotIn("test-key", msg["content"])

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_uses_bearer_auth(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "mode": "browsing", "operation": "none", "confidence": 0.5,
        }))
        self.parser.parse("hello", {"active_query": "", "turn": 1})
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")


class HybridIntentParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = IntentClassifier()
        self.deterministic = DeterministicIntentParser(self.classifier)

    def test_uses_llm_when_available(self) -> None:
        llm = MagicMock(spec=OpenAIIntentParser)
        llm.parse.return_value = IntentResult(
            mode="override", confidence=0.9, source="llm",
        )
        parser = HybridIntentParser(self.deterministic, llm)
        result = parser.parse(
            "On second thought, white would suit me better.",
            {"active_query": "black shoes"},
        )
        self.assertEqual(result.source, "llm")
        self.assertEqual(result.mode, "override")
        llm.parse.assert_called_once()

    def test_llm_handles_all_messages(self) -> None:
        """Even simulator-template-like messages go through the LLM."""
        llm = MagicMock(spec=OpenAIIntentParser)
        llm.parse.return_value = IntentResult(
            mode="buying", confidence=0.95, source="llm",
        )
        parser = HybridIntentParser(self.deterministic, llm)
        result = parser.parse(
            "I'm looking for shoes. A key requirement is: cotton.",
            {"active_query": ""},
        )
        self.assertEqual(result.source, "llm")
        llm.parse.assert_called_once()

    def test_low_confidence_falls_back(self) -> None:
        llm = MagicMock(spec=OpenAIIntentParser)
        llm.parse.return_value = IntentResult(
            mode="override", confidence=0.1, source="llm",
        )
        parser = HybridIntentParser(self.deterministic, llm)
        result = parser.parse("maybe change color?", {"active_query": ""})
        self.assertEqual(result.source, "deterministic")

    def test_llm_exception_falls_back(self) -> None:
        llm = MagicMock(spec=OpenAIIntentParser)
        llm.parse.side_effect = RuntimeError("API error")
        parser = HybridIntentParser(self.deterministic, llm)
        result = parser.parse("anything except leather", {"active_query": "shoes"})
        self.assertEqual(result.source, "deterministic")

    def test_no_llm_always_deterministic(self) -> None:
        parser = HybridIntentParser(self.deterministic, llm=None)
        result = parser.parse("something warm for winter", {"active_query": ""})
        self.assertEqual(result.source, "deterministic")

    def test_missing_key_falls_back(self) -> None:
        parser = HybridIntentParser(self.deterministic, llm=None)
        result = parser.parse(
            "On second thought, white would suit me better.",
            {"active_query": "black shoes"},
        )
        self.assertEqual(result.source, "deterministic")


class StateOperationsTest(unittest.TestCase):
    """Test that IntentResult fields map to correct state operations."""

    def test_add_constraints_preserved(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "add_constraints": {"material": "cotton", "color": "blue"},
        })
        assert result is not None
        self.assertEqual(result.add_constraints["material"], "cotton")
        self.assertEqual(result.add_constraints["color"], "blue")

    def test_remove_constraints_preserved(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "remove", "confidence": 0.8,
            "remove_constraints": ["black", "formal"],
        })
        assert result is not None
        self.assertEqual(result.remove_constraints, ["black", "formal"])

    def test_negative_constraints_preserved(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.85,
            "negative_constraints": ["leather", "suede"],
        })
        assert result is not None
        self.assertEqual(result.negative_constraints, ["leather", "suede"])

    def test_no_preference_preserved(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "none", "confidence": 0.9,
            "no_preference": ["color", "brand"],
        })
        assert result is not None
        self.assertEqual(result.no_preference, ["color", "brand"])

    def test_override_mode_replaces_state(self) -> None:
        result = validate_intent_result({
            "mode": "override", "operation": "replace", "confidence": 0.95,
            "add_constraints": {"color": "white"},
            "remove_constraints": ["black"],
        })
        assert result is not None
        self.assertEqual(result.mode, "override")
        self.assertEqual(result.operation, "replace")


class SessionIsolationTest(unittest.TestCase):
    """Verify that separate parser instances maintain independent caches."""

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_separate_instances_have_separate_caches(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "mode": "buying", "operation": "add", "confidence": 0.8,
        }))
        parser_a = OpenAIIntentParser(api_key="key-a", timeout=2.0)
        parser_b = OpenAIIntentParser(api_key="key-b", timeout=2.0)

        parser_a.parse("shoes", {"active_query": "", "turn": 1})
        self.assertEqual(len(parser_a._cache), 1)
        self.assertEqual(len(parser_b._cache), 0)


class SuggestedQuestionTest(unittest.TestCase):
    def test_valid_suggested_question_preserved(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "suggested_question": "material",
        })
        assert result is not None
        self.assertEqual(result.suggested_question, "material")

    def test_invalid_suggested_question_becomes_none(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "suggested_question": "weight",
        })
        assert result is not None
        self.assertIsNone(result.suggested_question)

    def test_null_suggested_question(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "suggested_question": None,
        })
        assert result is not None
        self.assertIsNone(result.suggested_question)

    def test_missing_suggested_question_defaults_to_none(self) -> None:
        result = validate_intent_result({
            "mode": "buying", "operation": "add", "confidence": 0.9,
        })
        assert result is not None
        self.assertIsNone(result.suggested_question)

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_llm_returns_suggested_question(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "suggested_question": "color",
        }))
        parser = OpenAIIntentParser(api_key="test-key", timeout=2.0)
        result = parser.parse("cotton shoes", {"active_query": "", "turn": 1})
        self.assertEqual(result.suggested_question, "color")


class CallOpenAITest(unittest.TestCase):
    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_shared_helper_returns_parsed_json(self, mock_urlopen) -> None:
        from starter.intent_parser import call_openai
        mock_urlopen.return_value = _mock_urlopen(_openai_response({
            "order": [2, 1, 3],
        }))
        parsed, pt, ct = call_openai("key", "system", "user", max_tokens=128)
        self.assertEqual(parsed, {"order": [2, 1, 3]})
        self.assertEqual(pt, 10)
        self.assertEqual(ct, 5)

    @patch("starter.intent_parser.urllib.request.urlopen")
    def test_shared_helper_returns_none_on_error(self, mock_urlopen) -> None:
        from starter.intent_parser import call_openai
        mock_urlopen.side_effect = TimeoutError("timed out")
        parsed, pt, ct = call_openai("key", "system", "user")
        self.assertIsNone(parsed)
        self.assertEqual(pt, 0)


class LoadApiKeyTest(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key-123"}, clear=False)
    def test_reads_from_environment(self) -> None:
        self.assertEqual(load_api_key(), "env-key-123")

    @patch.dict("os.environ", {}, clear=True)
    @patch("starter.intent_parser.Path.is_file", return_value=False)
    def test_returns_none_when_missing(self, _mock) -> None:
        self.assertIsNone(load_api_key())


if __name__ == "__main__":
    unittest.main()
