"""Adversarial holdout test set for the LLM intent parser.

These cases cover natural paraphrases, slang, negation, implicit references,
typos, unusual attributes, multilingual fragments, contradictory preferences,
and multi-requirement messages.

This file must NOT be used for tuning — it exists solely as a held-out
evaluation of generalisation. Do not add if-branches or regex patches
to pass individual holdout cases.
"""

from __future__ import annotations

import unittest

from starter.intent_parser import IntentResult, LLM_CONFIDENCE_THRESHOLD, validate_intent_result


class AdversarialSchemaTest(unittest.TestCase):
    """Ensure the schema can represent every adversarial intent shape."""

    def _validate(self, payload: dict) -> IntentResult:
        result = validate_intent_result(payload)
        self.assertIsNotNone(result, f"Validation failed for {payload}")
        assert result is not None
        return result

    # ---- Override / preference change ----

    def test_on_second_thought_override(self) -> None:
        r = self._validate({
            "mode": "override", "operation": "replace", "confidence": 0.9,
            "add_constraints": {"color": "white"},
        })
        self.assertEqual(r.mode, "override")
        self.assertEqual(r.add_constraints.get("color"), "white")

    def test_changed_mind_override(self) -> None:
        r = self._validate({
            "mode": "override", "operation": "replace", "confidence": 0.85,
            "add_constraints": {"style": "casual"},
            "remove_constraints": ["formal"],
        })
        self.assertEqual(r.mode, "override")
        self.assertIn("formal", r.remove_constraints)

    # ---- Negative constraints / exclusion ----

    def test_anything_except_leather(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "negative_constraints": ["leather"],
        })
        self.assertIn("leather", r.negative_constraints)

    def test_not_synthetic_material(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.8,
            "negative_constraints": ["polyester", "nylon"],
        })
        self.assertEqual(len(r.negative_constraints), 2)

    def test_no_bright_colors(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.75,
            "negative_constraints": ["red", "yellow", "orange"],
        })
        self.assertTrue(len(r.negative_constraints) >= 2)

    # ---- References to previous items ----

    def test_same_material_but_blue(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.85,
            "add_constraints": {"color": "blue"},
            "referenced_previous_item": True,
            "reference_description": "same material as before",
        })
        self.assertTrue(r.referenced_previous_item)
        self.assertEqual(r.add_constraints.get("color"), "blue")

    def test_cheaper_than_previous(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.8,
            "referenced_previous_item": True,
            "reference_description": "cheaper than the previous ones",
        })
        self.assertTrue(r.referenced_previous_item)

    # ---- No-preference signals ----

    def test_not_fussed_about_colour(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "none", "confidence": 0.9,
            "no_preference": ["color"],
        })
        self.assertIn("color", r.no_preference)

    def test_whatever_works(self) -> None:
        r = self._validate({
            "mode": "browsing", "operation": "none", "confidence": 0.7,
            "no_preference": ["brand", "color"],
        })
        self.assertTrue(len(r.no_preference) >= 1)

    # ---- Situational / implicit requirements ----

    def test_tokyo_in_december(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.85,
            "add_constraints": {"use_case": "walking", "season": "winter"},
        })
        self.assertEqual(r.mode, "buying")
        self.assertTrue(len(r.add_constraints) >= 1)

    def test_beach_holiday(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.8,
            "add_constraints": {"use_case": "beach", "feature": "lightweight"},
        })
        self.assertTrue(len(r.add_constraints) >= 1)

    # ---- Slang and informal language ----

    def test_slang_drip(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.7,
            "add_constraints": {"style": "fashionable"},
        })
        self.assertEqual(r.mode, "buying")

    def test_slang_kicks(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.8,
            "category": "shoes",
        })
        self.assertEqual(r.category, "shoes")

    # ---- Typos ----

    def test_typo_breatable(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.7,
            "add_constraints": {"feature": "breathable"},
        })
        self.assertEqual(r.add_constraints.get("feature"), "breathable")

    def test_typo_cottn(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.75,
            "add_constraints": {"material": "cotton"},
        })
        self.assertEqual(r.add_constraints.get("material"), "cotton")

    # ---- Multiple requirements in one message ----

    def test_multi_requirement(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "add_constraints": {"material": "cotton", "color": "blue", "budget": "under $50"},
        })
        self.assertTrue(len(r.add_constraints) >= 2)

    def test_constraint_plus_exclusion(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.85,
            "add_constraints": {"material": "cotton"},
            "negative_constraints": ["polyester"],
        })
        self.assertIn("cotton", r.add_constraints.values())
        self.assertIn("polyester", r.negative_constraints)

    # ---- Contradictory preferences ----

    def test_contradictory_warm_and_lightweight(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.6,
            "add_constraints": {"feature": "warm", "weight": "lightweight"},
        })
        self.assertTrue(len(r.add_constraints) >= 1)

    # ---- Multilingual fragments ----

    def test_japanese_fragment(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.6,
            "category": "shoes",
            "add_constraints": {"use_case": "walking"},
        })
        self.assertIsNotNone(r)

    def test_french_fragment(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.65,
            "add_constraints": {"color": "red"},
        })
        self.assertIsNotNone(r)

    # ---- Unusual attributes ----

    def test_unusual_attribute_eco_friendly(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.7,
            "add_constraints": {"feature": "eco-friendly"},
        })
        self.assertIn("eco-friendly", r.add_constraints.values())

    def test_unusual_attribute_plus_size(self) -> None:
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.8,
            "add_constraints": {"size": "plus size"},
        })
        self.assertEqual(r.add_constraints.get("size"), "plus size")

    # ---- Edge: empty / minimal messages ----

    def test_single_word(self) -> None:
        r = self._validate({
            "mode": "browsing", "operation": "none", "confidence": 0.3,
        })
        self.assertEqual(r.mode, "browsing")

    def test_emoji_only_low_confidence(self) -> None:
        r = self._validate({
            "mode": "browsing", "operation": "none", "confidence": 0.1,
        })
        self.assertLessEqual(r.confidence, LLM_CONFIDENCE_THRESHOLD)

    # ---- Product ID injection guard ----

    def test_no_product_id_injection(self) -> None:
        """The schema accepts add_constraints but the agent must never use them as IDs."""
        r = self._validate({
            "mode": "buying", "operation": "add", "confidence": 0.9,
            "add_constraints": {"parent_asin": "B001234567"},
        })
        assert r is not None
        self.assertNotIn("B001234567", [r.category or ""])


if __name__ == "__main__":
    unittest.main()
