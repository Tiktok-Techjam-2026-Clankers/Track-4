from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import (
    Agent,
    ConversationMemory,
    HybridRanker,
    InMemoryVectorIndex,
    IntentCardIndex,
    IntentClassifier,
    _catalog_constraints,
)


class IntentAndMemoryTest(unittest.TestCase):
    def test_intent_classifier_covers_all_three_routes(self) -> None:
        classifier = IntentClassifier()
        self.assertEqual(classifier.classify("I am exploring shoes"), "browsing")
        self.assertEqual(
            classifier.classify("black leather running shoes under $80"),
            "buying",
        )
        self.assertEqual(
            classifier.classify("Actually, make them white instead"),
            "override",
        )

    def test_override_replaces_active_retrieval_context(self) -> None:
        memory = ConversationMemory({})
        classifier = IntentClassifier()
        memory.observe("black running shoes", classifier)
        memory.observe("Actually, make them white casual sneakers", classifier)

        self.assertNotIn("black", memory.query())
        self.assertIn("white casual sneakers", memory.query())
        self.assertEqual(memory.intent, "override")
        self.assertEqual(len(memory.previous_intents), 1)

    def test_boundary_reply_is_remembered_but_not_searched(self) -> None:
        memory = ConversationMemory({})
        classifier = IntentClassifier()
        memory.observe("walking shoes", classifier)
        memory.last_question = "color"
        memory.observe("I don't have a preference; use your judgment", classifier)

        self.assertEqual(memory.query(), "walking shoes")
        self.assertIn("color", memory.declined_attributes)
        self.assertEqual(len(memory.history), 2)

    def test_conversation_memories_are_isolated(self) -> None:
        classifier = IntentClassifier()
        first = ConversationMemory({"summary": "first"})
        second = ConversationMemory({"summary": "second"})

        first.observe("black running shoes", classifier)
        second.observe("blue cotton shirt", classifier)

        self.assertEqual(first.query(), "black running shoes")
        self.assertEqual(second.query(), "blue cotton shirt")
        self.assertNotEqual(first.history, second.history)

    def test_override_resets_the_recommendation_ladder(self) -> None:
        classifier = IntentClassifier()
        memory = ConversationMemory({})
        memory.observe("black running shoes", classifier)
        memory.recommendation_ladder = ["A", "B"]
        memory.ladder_position = 1

        memory.observe("Actually, make them white casual sneakers", classifier)

        self.assertEqual(memory.recommendation_ladder, [])
        self.assertEqual(memory.ladder_position, 0)


class RetrievalAndRankingTest(unittest.TestCase):
    def test_semantic_aliases_share_vector_space(self) -> None:
        index = InMemoryVectorIndex(
            ["shoe", "dress", "jacket"],
            ["comfortable sneakers for running", "formal silk gown", "winter coat"],
        )
        self.assertEqual(index.search("comfy trainers", limit=1), ["shoe"])

    def test_fusion_is_unique_and_intent_weighted(self) -> None:
        buying = HybridRanker.fuse(
            ["lexical", "shared"], ["semantic", "shared"],
            ["shared", "lexical", "semantic"], ["shared", "semantic", "lexical"],
            "buying", 3,
        )
        browsing = HybridRanker.fuse(
            ["lexical", "shared"], ["semantic", "shared"],
            ["shared", "lexical", "semantic"], ["shared", "semantic", "lexical"],
            "browsing", 3,
        )
        self.assertEqual(len({item for item, _ in buying}), len(buying))
        self.assertEqual(buying[0][0], "shared")
        self.assertEqual(browsing[0][0], "shared")
        self.assertNotEqual(buying[0][1], browsing[0][1])

    def test_ordered_prefix_outweighs_popularity(self) -> None:
        index = IntentCardIndex(
            ["blue", "red"],
            [
                ["cotton", "color blue", "pull closure"],
                ["cotton", "color red", "pull closure"],
            ],
        )
        ranked = index.prefix_search(
            [
                "A key requirement is: cotton.",
                "For that, what matters is: color: blue.",
            ],
            ["blue", "red"],
            {"blue": 0.0, "red": 100.0},
        )
        self.assertEqual(ranked[0], "blue")

    def test_override_reconciles_old_and_new_card_constraints(self) -> None:
        index = IntentCardIndex(
            ["target", "other"],
            [
                ["cotton", "color blue", "pull closure", "machine wash"],
                ["cotton", "color blue", "pull closure", "dry clean"],
            ],
        )
        ranked = index.override_search(
            ["Actually, what I need is: cotton."],
            "I'm looking for shirts. Machine wash. "
            "For that, what matters is: cotton; color blue.",
            ["target", "other"],
            {"target": 0.0, "other": 100.0},
        )
        self.assertEqual(ranked[0], "target")

    def test_catalog_constraints_preserve_unicode_clauses(self) -> None:
        product = {
            "features": ["进口", "Plastic frame"],
            "details": {},
            "price": None,
        }
        self.assertEqual(
            _catalog_constraints(product, "进口 black sunglasses")[:2],
            ["进口", "color black"],
        )


class AgentContractTest(unittest.TestCase):
    def test_agent_returns_ranked_contract_payload(self) -> None:
        products = [
            {
                "parent_asin": "A",
                "title": "Blue comfortable running shoe",
                "categories": ["Shoes"],
                "features": ["breathable"],
                "details": {},
                "description": [],
                "store": "Example",
            },
            {
                "parent_asin": "B",
                "title": "Red formal silk dress",
                "categories": ["Dresses"],
                "features": ["evening"],
                "details": {},
                "description": [],
                "store": "Example",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            agent = Agent(catalog)
            agent.reset("session", {})
            response = agent.respond("session", "comfy blue trainers", 1, 2)

        # The turn-aware policy intentionally reveals one candidate early,
        # then expands the stable Top-10 ladder on later turns.
        self.assertEqual(len(response["recommendations"]), 1)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A")
        self.assertIn(response["ask_attribute"], {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other",
        })
        self.assertEqual(response["usage"], {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        })


if __name__ == "__main__":
    unittest.main()
