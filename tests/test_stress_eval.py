from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    customer_reply as official_customer_reply,
    evaluate,
    initial_message as official_initial_message,
    intent_card,
)
from stress.contract import check_response
from stress.personas import PERSONAS, paraphrase_constraint, terse_constraint
from stress.runner import customer_reply, opening_message, run_persona

from scripts.run_stress_eval import REPO_ROOT, relative_label, stratified_sample


CATALOG_ROWS = [
    {
        "parent_asin": "A",
        "title": "Blue running shoe",
        "features": ["cotton"],
        "details": {"department": "womens"},
        "description": ["walking shoe"],
        "categories": ["Clothing", "Shoes"],
        "store": "Example",
        "average_rating": 4.2,
        "rating_number": 10,
        "price": 49.0,
    },
    {
        "parent_asin": "B",
        "title": "Black winter boot",
        "features": ["leather"],
        "details": {"department": "womens"},
        "description": ["winter boot"],
        "categories": ["Clothing", "Boots"],
        "store": "Example",
        "average_rating": 4.4,
        "rating_number": 12,
        "price": 89.0,
    },
]

SAMPLES = [
    {
        "sample_id": "stress_0001",
        "scenario_type": "buying",
        "user_profile": {"summary": "x"},
        "ground_truth": {"parent_asin": "A"},
    },
    {
        "sample_id": "stress_0002",
        "scenario_type": "browsing",
        "user_profile": {"summary": "y"},
        "ground_truth": {"parent_asin": "B"},
    },
]


class ColorSeekingAgent:
    """Asks for a colour, then converts once the customer has mentioned one."""

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.seen: list[str] = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.seen.append(user_message.lower())
        corpus = " ".join(self.seen)
        recommendations = []
        if "blue" in corpus:
            recommendations = [{"parent_asin": "A"}]
        elif "black" in corpus:
            recommendations = [{"parent_asin": "B"}]
        return {
            "message": "What colour would you like?",
            "ask_attribute": "color",
            "recommendations": recommendations,
        }


class StressHarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        catalog_path = Path(cls._directory.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8"
        )
        cls.catalog_ids, cls.categories, cls.products = catalog_index(catalog_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _card_sample(self) -> dict:
        card = intent_card(self.products["A"])
        return {**SAMPLES[0], "intent_card": card, "behavior": {"scenario_type": "buying"}}

    def test_verbatim_persona_matches_official_opening_wording(self) -> None:
        sample = self._card_sample()
        official_disclosed: set[str] = set()
        stress_disclosed: set[str] = set()
        expected = official_initial_message(sample, "Shoes", official_disclosed)
        actual = opening_message(
            PERSONAS["verbatim"], sample, "Shoes", stress_disclosed, random.Random(0)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(stress_disclosed, official_disclosed)

    def test_verbatim_persona_matches_official_reply_wording(self) -> None:
        sample = self._card_sample()
        for attribute in ("color", "material", "budget", "size", None, "not_an_attribute"):
            with self.subTest(attribute=attribute):
                official_disclosed: set[str] = set()
                stress_disclosed: set[str] = set()
                expected, expected_boundary = official_customer_reply(
                    sample, attribute, official_disclosed, False
                )
                actual, actual_boundary = customer_reply(
                    PERSONAS["verbatim"], sample, attribute, stress_disclosed, False, random.Random(0)
                )
                self.assertEqual(actual, expected)
                self.assertEqual(actual_boundary, expected_boundary)
                self.assertEqual(stress_disclosed, official_disclosed)

    def test_verbatim_persona_reproduces_official_evaluator_metrics(self) -> None:
        official = evaluate(
            ColorSeekingAgent(), SAMPLES, self.catalog_ids, self.categories, self.products
        )
        stress = run_persona(
            ColorSeekingAgent(), PERSONAS["verbatim"], SAMPLES,
            self.catalog_ids, self.categories, self.products,
        )
        for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score"):
            self.assertEqual(stress[key], official[key], key)
        self.assertEqual(
            [(item["sample_id"], item["first_hit_turn"], item["best_rank"]) for item in stress["sessions"]],
            [(item["sample_id"], item["first_hit_turn"], item["best_rank"]) for item in official["sessions"]],
        )

    def test_every_persona_run_is_reproducible(self) -> None:
        for name, persona in PERSONAS.items():
            with self.subTest(persona=name):
                first = run_persona(
                    ColorSeekingAgent(), persona, SAMPLES,
                    self.catalog_ids, self.categories, self.products, transcripts=True,
                )
                second = run_persona(
                    ColorSeekingAgent(), persona, SAMPLES,
                    self.catalog_ids, self.categories, self.products, transcripts=True,
                )
                self.assertEqual(first, second)

    def test_personas_disclose_the_same_constraints_in_the_same_order(self) -> None:
        sample = self._card_sample()
        baseline: set[str] | None = None
        for name, persona in PERSONAS.items():
            with self.subTest(persona=name):
                disclosed: set[str] = set()
                for attribute in ("color", "material", "budget", "style"):
                    customer_reply(persona, sample, attribute, disclosed, False, random.Random(1))
                if baseline is None:
                    baseline = disclosed
                self.assertEqual(disclosed, baseline)

    def test_non_control_personas_reword_without_dropping_the_signal(self) -> None:
        sample = self._card_sample()
        verbatim, _ = customer_reply(
            PERSONAS["verbatim"], sample, "color", set(), False, random.Random(2)
        )
        for name in ("paraphrase", "terse", "terse_paraphrase"):
            with self.subTest(persona=name):
                reworded, _ = customer_reply(
                    PERSONAS[name], sample, "color", set(), False, random.Random(2)
                )
                self.assertNotEqual(reworded, verbatim)
                self.assertIn("blue", reworded.lower())

    def test_paraphrase_rewrites_key_value_details_and_budgets(self) -> None:
        rng = random.Random(3)
        self.assertEqual(paraphrase_constraint("Department: womens", rng), "made for women")
        self.assertEqual(paraphrase_constraint("color: blue", rng), "in blue")
        budget = paraphrase_constraint("budget around $49.0", rng)
        self.assertIn("$49", budget)
        self.assertNotIn("budget around", budget)

    def test_terse_keeps_material_and_colour_tokens(self) -> None:
        terse = terse_constraint("a very comfortable and breathable long sleeve cotton shirt in black")
        self.assertIn("cotton", terse)
        self.assertIn("black", terse)
        self.assertNotIn(" and ", f" {terse} ")

    def test_contract_check_flags_known_violations(self) -> None:
        self.assertEqual(check_response({"message": "hi", "ask_attribute": None, "recommendations": []}, {"A"}), [])
        violations = check_response(
            {
                "message": 7,
                "ask_attribute": "vibe",
                "recommendations": [{"parent_asin": "A"}, {"parent_asin": "A"}, {"parent_asin": "Z"}],
                "usage": {"prompt_tokens": -1, "completion_tokens": 2},
                "extra": True,
            },
            {"A"},
        )
        self.assertIn("message_not_string", violations)
        self.assertIn("ask_attribute_not_allowed", violations)
        self.assertIn("recommendation_duplicate", violations)
        self.assertIn("recommendation_not_in_catalog", violations)
        self.assertIn("usage_prompt_tokens_negative", violations)
        self.assertIn("response_unknown_key:extra", violations)

    def test_relative_label_accepts_paths_outside_the_repo(self) -> None:
        inside = REPO_ROOT / "data" / "private_set.jsonl"
        self.assertEqual(relative_label(inside), str(Path("data") / "private_set.jsonl"))
        outside = Path(tempfile.gettempdir()).resolve() / "elsewhere.jsonl"
        self.assertEqual(relative_label(outside), str(outside))

    def test_stratified_sample_is_deterministic_and_keeps_the_scenario_mix(self) -> None:
        samples = [
            {"sample_id": f"s{index}", "scenario_type": scenario}
            for index, scenario in enumerate(["buying"] * 8 + ["browsing"] * 8 + ["boundary"] * 4)
        ]
        picked = stratified_sample(samples, 6)
        self.assertEqual(len(picked), 6)
        self.assertEqual(picked, stratified_sample(samples, 6))
        self.assertEqual({item["scenario_type"] for item in picked}, {"buying", "browsing", "boundary"})
        self.assertEqual(stratified_sample(samples, 100), samples)


if __name__ == "__main__":
    unittest.main()
