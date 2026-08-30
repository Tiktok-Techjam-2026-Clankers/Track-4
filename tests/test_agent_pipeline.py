from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_intent_classifier_fallback_returns_browsing(self) -> None:
        classifier = IntentClassifier()
        self.assertEqual(classifier.classify("I am exploring shoes"), "browsing")
        self.assertEqual(classifier.classify("black leather running shoes under $80"), "browsing")
        self.assertEqual(classifier.classify("Actually, make them white instead"), "browsing")

    def test_override_replaces_active_retrieval_context(self) -> None:
        memory = ConversationMemory({})
        classifier = IntentClassifier()
        memory.observe("black running shoes", classifier)
        memory.observe("Actually, make them white casual sneakers", classifier, mode_override="override")

        self.assertNotIn("black", memory.query())
        self.assertIn("white casual sneakers", memory.query())
        self.assertEqual(memory.intent, "override")
        self.assertEqual(len(memory.previous_intents), 1)

    def test_override_phrases_replace_stale_requirements(self) -> None:
        classifier = IntentClassifier()
        for phrase in (
            "Change of plan: white casual sneakers",
            "Scrap that; white casual sneakers",
            "Let me correct myself: white casual sneakers",
        ):
            with self.subTest(phrase=phrase):
                memory = ConversationMemory({})
                memory.observe("black running shoes", classifier)
                memory.observe(phrase, classifier, mode_override="override")
                self.assertEqual(memory.intent, "override")
                self.assertNotIn("black", memory.query())

    def test_boundary_reply_is_remembered_but_not_searched(self) -> None:
        memory = ConversationMemory({})
        classifier = IntentClassifier()
        memory.observe("walking shoes", classifier)
        memory.last_question = "color"
        memory.observe(
            "I don't have a preference; use your judgment",
            classifier,
            no_pref_override=True,
        )

        self.assertEqual(memory.query(), "walking shoes")
        self.assertIn("color", memory.declined_attributes)
        self.assertEqual(len(memory.history), 2)

    def test_no_preference_reply_is_not_added_to_search(self) -> None:
        memory = ConversationMemory({})
        classifier = IntentClassifier()
        memory.observe("walking shoes", classifier)
        memory.last_question = "color"
        memory.observe("No strong feelings on colour", classifier, no_pref_override=True)

        self.assertEqual(memory.query(), "walking shoes")
        self.assertIn("color", memory.declined_attributes)

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

        memory.observe(
            "Actually, make them white casual sneakers",
            classifier,
            mode_override="override",
        )

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

    def test_fuzzy_cards_match_natural_semantic_phrases(self) -> None:
        index = IntentCardIndex(
            ["target", "other"],
            [
                ["waterresistant", "insulated", "adjustable"],
                ["silk", "formal", "dry clean"],
            ],
        )
        ranked = index.fuzzy_search(
            ["Something that keeps water out, keeps heat in, and is easy to adjust"],
            ["target", "other"],
            {"target": 0.0, "other": 100.0},
            set(),
            limit=2,
        )
        self.assertEqual(ranked[0], "target")

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
            with patch("starter.agent.load_api_key", return_value=None):
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


class ComputeWeightsTest(unittest.TestCase):
    def test_default_buying_weights(self) -> None:
        weights, k = HybridRanker.compute_weights("buying")
        self.assertEqual(weights, (0.25, 0.10, 0.50, 0.40))
        self.assertEqual(k, 20)

    def test_late_turn_boosts_evidence(self) -> None:
        weights, _ = HybridRanker.compute_weights("buying", turn=6)
        base = HybridRanker.WEIGHTS["buying"]
        self.assertGreater(weights[2], base[2])
        self.assertLess(weights[0], base[0])

    def test_many_constraints_shrinks_k(self) -> None:
        _, k = HybridRanker.compute_weights("buying", constraint_count=4)
        self.assertLess(k, HybridRanker.FUSION_K["buying"])

    def test_post_override_boosts_semantic(self) -> None:
        weights, _ = HybridRanker.compute_weights(
            "override", is_post_override=True, phase_turn=1,
        )
        base = HybridRanker.WEIGHTS["override"]
        self.assertGreater(weights[1], base[1])

    def test_preference_tags_boost_semantic(self) -> None:
        weights, _ = HybridRanker.compute_weights("browsing", has_preference_tags=True)
        base = HybridRanker.WEIGHTS["browsing"]
        self.assertGreater(weights[1], base[1])

    def test_weights_clamped_to_valid_range(self) -> None:
        weights, k = HybridRanker.compute_weights(
            "buying", turn=8, constraint_count=10,
            is_post_override=True, phase_turn=1, has_preference_tags=True,
        )
        for w in weights:
            self.assertGreaterEqual(w, 0.05)
            self.assertLessEqual(w, 1.0)
        self.assertGreaterEqual(k, 5)

    def test_unknown_intent_uses_browsing(self) -> None:
        weights, _ = HybridRanker.compute_weights("nonexistent")
        base = HybridRanker.WEIGHTS["browsing"]
        self.assertEqual(weights, base)


class FuseWithDynamicWeightsTest(unittest.TestCase):
    def test_fuse_accepts_explicit_weights_and_k(self) -> None:
        result = HybridRanker.fuse(
            ["a", "b"], ["b", "a"], ["a"], ["b"],
            "buying", 3,
            weights=(1.0, 0.05, 0.05, 0.05), fusion_k=10,
        )
        self.assertEqual(result[0][0], "a")

    def test_fuse_none_weights_uses_defaults(self) -> None:
        result = HybridRanker.fuse(
            ["a", "b"], ["b", "a"], ["a"], ["b"],
            "buying", 3,
            weights=None, fusion_k=None,
        )
        self.assertEqual(len(result), 2)


class LLMRerankerTest(unittest.TestCase):
    def test_apply_order_remaps_correctly(self) -> None:
        from starter.agent import LLMReranker
        head = [("A", 1.0), ("B", 0.9), ("C", 0.8)]
        tail = [("D", 0.5)]
        result = LLMReranker._apply_order([3, 1, 2], head, tail)
        self.assertEqual([r[0] for r in result], ["C", "A", "B", "D"])

    def test_apply_order_appends_missing_indices(self) -> None:
        from starter.agent import LLMReranker
        head = [("A", 1.0), ("B", 0.9), ("C", 0.8)]
        result = LLMReranker._apply_order([2], head, [])
        self.assertEqual([r[0] for r in result], ["B", "A", "C"])

    def test_apply_order_ignores_out_of_range(self) -> None:
        from starter.agent import LLMReranker
        head = [("A", 1.0), ("B", 0.9)]
        result = LLMReranker._apply_order([99, 1], head, [])
        self.assertEqual([r[0] for r in result], ["A", "B"])

    def test_rerank_returns_original_on_single_item(self) -> None:
        from starter.agent import LLMReranker
        reranker = LLMReranker("key", {"A": "shoes"})
        result, pt, ct = reranker.rerank([("A", 1.0)], "shoes", {})
        self.assertEqual(result, [("A", 1.0)])
        self.assertEqual(pt, 0)

    @patch("starter.agent.call_openai")
    def test_rerank_reorders_on_success(self, mock_call) -> None:
        from starter.agent import LLMReranker
        mock_call.return_value = ({"order": [2, 1]}, 50, 20)
        reranker = LLMReranker("key", {"A": "red shoes", "B": "blue shoes"})
        result, pt, ct = reranker.rerank(
            [("A", 1.0), ("B", 0.9)], "blue shoes", {},
        )
        self.assertEqual(result[0][0], "B")
        self.assertEqual(pt, 50)

    @patch("starter.agent.call_openai")
    def test_rerank_falls_back_on_api_failure(self, mock_call) -> None:
        from starter.agent import LLMReranker
        mock_call.return_value = (None, 0, 0)
        reranker = LLMReranker("key", {"A": "shoes", "B": "boots"})
        original = [("A", 1.0), ("B", 0.9)]
        result, _, _ = reranker.rerank(original, "shoes", {})
        self.assertEqual(result, original)


class ChooseQuestionTest(unittest.TestCase):
    def test_llm_suggestion_takes_priority(self) -> None:
        memory = ConversationMemory({})
        memory.suggested_question = "material"
        attribute = memory.choose_question()
        self.assertEqual(attribute, "material")

    def test_llm_suggestion_skipped_if_declined(self) -> None:
        memory = ConversationMemory({})
        memory.suggested_question = "color"
        memory.declined_attributes.add("color")
        attribute = memory.choose_question()
        self.assertEqual(attribute, "other")

    def test_llm_suggestion_skipped_if_already_asked(self) -> None:
        memory = ConversationMemory({})
        memory.suggested_question = "style"
        memory.asked_counts["style"] = 1
        attribute = memory.choose_question()
        self.assertEqual(attribute, "other")

    def test_falls_back_to_sequence_after_open_limit(self) -> None:
        memory = ConversationMemory({})
        memory.asked_counts["other"] = 3
        attribute = memory.choose_question()
        self.assertNotEqual(attribute, "other")


if __name__ == "__main__":
    unittest.main()
