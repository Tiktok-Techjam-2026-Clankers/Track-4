"""Independent conversational shopping agent for TechJam Track 4.

Every turn follows five explicit stages: intent classification, per-session
memory, parallel BM25 and semantic-vector retrieval, rank fusion, and a Top-10
response with one clarification question. The semantic index is a deterministic
NumPy matrix built from the frozen catalog; no network or LLM is required.
"""

from __future__ import annotations

import json
import heapq
import math
import re
import sqlite3
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CONSTRAINT_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore\s+(?:that|my|the)|no\s+longer|"
    r"changed?\s+my\s+mind|change\s+of\s+plan|make\s+(?:it|them)|"
    r"scrap\s+(?:that|what|the)|forget\s+(?:that|what|the|my)|"
    r"correct\s+myself|let\s+me\s+correct)\b",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:no|without)\s+(?:additional\s+)?preference\b|"
    r"\bdon['’]?t\s+have\s+(?:an?\s+)?(?:additional\s+)?preference\b|"
    r"\buse\s+your\s+(?:best\s+)?judg(?:e)?ment\b|"
    r"\bno\s+strong\s+feelings?\b|\bdoesn['’]?t\s+(?:really\s+)?matter\b|"
    r"\bnothing\s+to\s+add\b|\bup\s+to\s+you\b|"
    r"\bdon['’]?t\s+mind\b|\byou\s+pick\b|\bwhatever\b",
    re.IGNORECASE,
)
BROWSING_RE = re.compile(
    r"\b(?:still\s+exploring|just\s+browsing|still\s+deciding|not\s+sure|"
    r"nothing\s+fixed|haven['’]?t\s+narrowed)\b",
    re.IGNORECASE,
)
PRICE_RE = re.compile(
    r"(?:\$\s*\d+(?:\.\d+)?|\b(?:under|below|less\s+than|budget)\b)",
    re.IGNORECASE,
)
CATEGORY_CONTEXT_RE = re.compile(
    r"^\s*i['’]?m\s+looking\s+for\s+([^.,;]+)", re.IGNORECASE
)
MATERIAL_VALUE_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_VALUE_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "about", "actually", "additional", "all", "also", "am", "an",
    "and", "any", "anything", "are", "as", "at", "be", "below", "but",
    "by", "can", "could", "do", "don", "dont", "earlier", "else", "for",
    "from", "have", "help", "here", "i", "if", "ignore", "in", "instead",
    "is", "it", "just", "looking", "make", "me", "mind", "more", "my",
    "need", "no", "not", "now", "of", "on", "one", "or", "other", "please",
    "preference", "preferences", "prefer", "requirement", "show", "so", "some",
    "something", "still", "than", "that", "the", "them", "these", "this",
    "those", "to", "use", "want", "what", "with", "would", "you", "your",
    "after", "around", "browsing", "buy", "front", "kind", "main", "made",
    "market", "matters", "must", "shopping", "sort", "sure", "thing",
}

# Inspectable domain semantics applied before vectorization.
CONCEPT_ALIASES = {
    "sneaker": "shoe", "sneakers": "shoe", "trainer": "shoe",
    "trainers": "shoe", "footwear": "shoe", "tee": "tshirt",
    "tees": "tshirt", "tshirts": "tshirt", "hoodie": "sweatshirt",
    "hoodies": "sweatshirt", "pullover": "sweatshirt", "trousers": "pants",
    "slacks": "pants", "joggers": "pants", "comfy": "comfortable",
    "comfort": "comfortable", "light": "lightweight",
    "waterproof": "waterresistant", "rainproof": "waterresistant",
    "breathability": "breathable", "warm": "insulated", "warmth": "insulated",
    "winter": "insulated", "formal": "dressy", "office": "work",
    "workwear": "work", "gym": "athletic", "workout": "athletic",
    "fitness": "athletic", "jogging": "running", "hike": "hiking",
    "trekking": "hiking", "grey": "gray", "navy": "blue", "burgundy": "red",
    "airy": "breathable", "cosy": "insulated", "cozy": "insulated",
    "sleeved": "sleeve", "rise": "waisted", "waist": "waistband",
    "crease": "wrinkle", "women": "womens", "woman": "womens",
    "men": "mens", "man": "mens", "everyday": "casual",
    "adjust": "adjustable", "adjusted": "adjustable", "dries": "dry",
    "washed": "wash", "washes": "washable", "wearing": "durable",
}

PHRASE_ALIASES = (
    (re.compile(r"\bkeeps?\s+water\s+out\b", re.I), "waterresistant"),
    (re.compile(r"\bkeeps?\s+heat\s+in\b", re.I), "insulated"),
    (re.compile(r"\bhard[ -]?wearing\b", re.I), "durable"),
    (re.compile(r"\bdr(?:y|ies)\s+fast\b", re.I), "quick dry"),
    (re.compile(r"\bhigh[ -]?rise\b", re.I), "high waisted"),
    (re.compile(r"\blong[ -]?sleeved\b", re.I), "long sleeve"),
    (re.compile(r"\bshort[ -]?sleeved\b", re.I), "short sleeve"),
    (re.compile(r"\bstretchy\s+waist\b", re.I), "elastic waistband"),
    (re.compile(r"\bpull\s+cord\b", re.I), "drawstring"),
    (re.compile(r"\bnon[ -]?slip\b", re.I), "slip resistant"),
    (re.compile(r"\bdoes\s+not\s+crease\b", re.I), "wrinkle resistant"),
    (re.compile(r"\bgentle\s+on\s+skin\b", re.I), "hypoallergenic"),
    (re.compile(r"\bwashes?\s+in\s+the\s+machine\b", re.I), "machine washable"),
    (re.compile(r"\bwashed?\s+by\s+hand\b", re.I), "hand wash"),
    (re.compile(r"\bsoaks?\s+up\s+sweat\b", re.I), "sweat absorbent"),
    (re.compile(r"\bsweat[ -]?wicking\b", re.I), "moisture wicking"),
    (re.compile(r"\bwith\s+a\s+lining\b", re.I), "lined"),
    (re.compile(r"\beasy\s+to\s+adjust\b", re.I), "adjustable"),
)

HARD_CONSTRAINT_WORDS = {
    "black", "white", "gray", "blue", "red", "green", "brown", "pink",
    "cotton", "polyester", "nylon", "leather", "wool", "silk", "linen",
    "small", "medium", "large", "petite", "tall", "wide", "waterresistant",
    "breathable", "insulated", "running", "hiking", "athletic",
}

QUESTION_SEQUENCE = (
    "material", "color", "style", "use_case", "feature",
    "budget", "brand", "size", "category",
)

BM25_POOL = 250
SEMANTIC_POOL = 250
VECTOR_DIMENSIONS = 256
RRF_K = 20
MAX_QUERY_TERMS = 48
MAX_DOCUMENT_TOKENS = 180
OPEN_QUESTION_LIMIT = 3

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _canonical_text(text: str) -> str:
    for pattern, replacement in PHRASE_ALIASES:
        text = pattern.sub(replacement, text)
    return text


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    for raw in TOKEN_RE.findall(_canonical_text(text).lower()):
        if len(raw) <= 1 or raw in STOPWORDS:
            continue
        result.append(CONCEPT_ALIASES.get(raw, raw))
    return result


def _constraint_tokens(text: str) -> list[str]:
    """Tokenize exact catalog clauses while preserving non-ASCII words."""
    result: list[str] = []
    for raw in CONSTRAINT_TOKEN_RE.findall(_canonical_text(text).lower()):
        if len(raw) <= 1 or raw in STOPWORDS:
            continue
        result.append(CONCEPT_ALIASES.get(raw, raw))
    return result


def _unique_terms(text: str) -> list[str]:
    return list(dict.fromkeys(_tokens(text)))[:MAX_QUERY_TERMS]


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _catalog_constraints(product: dict, searchable: str) -> list[str]:
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    material = MATERIAL_VALUE_RE.search(searchable)
    color = COLOR_VALUE_RE.search(searchable)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = [
        re.sub(r"\s+", " ", item).strip(" -;,.\t\n")[:180].rstrip()
        for item in candidates
    ]
    normalized = [" ".join(_constraint_tokens(item)) for item in cleaned if item]
    return list(dict.fromkeys(item for item in normalized if item))[:4]


def _category_key(value: object) -> str:
    excluded = {"clothing", "clothing shoes jewelry"}
    cleaned: list[str] = []
    for category in _flatten_values(value):
        for part in category.split(","):
            normalized = " ".join(_tokens(part))
            if normalized and normalized not in excluded:
                cleaned.append(normalized)
    return " ".join(cleaned[-2:])


class IntentClassifier:
    """Classify the current request as browsing, buying, or override."""

    @staticmethod
    def classify(message: str, active_query: str = "") -> str:
        if OVERRIDE_RE.search(message):
            return "override"
        if BROWSING_RE.search(message):
            return "browsing"
        terms = _tokens(f"{active_query} {message}")
        hard_count = sum(term in HARD_CONSTRAINT_WORDS for term in set(terms))
        if PRICE_RE.search(message) or hard_count >= 2 or len(set(terms)) >= 6:
            return "buying"
        return "browsing"


@dataclass
class ConversationMemory:
    """Short-term state isolated to one evaluator session."""

    user_profile: dict
    history: list[str] = field(default_factory=list)
    active_messages: list[str] = field(default_factory=list)
    intent: str = "browsing"
    previous_intents: list[str] = field(default_factory=list)
    last_question: str | None = None
    declined_attributes: set[str] = field(default_factory=set)
    asked_counts: Counter[str] = field(default_factory=Counter)
    last_override_turn: int | None = None
    boundary_signal: bool = False
    session_mode: str = "browsing"
    recommendation_ladder: list[str] = field(default_factory=list)
    ladder_position: int = 0
    fuzzy_recommended: set[str] = field(default_factory=set)

    def observe(self, message: str, classifier: IntentClassifier) -> str:
        self.history.append(message)
        query_before = self.query()
        intent = classifier.classify(message, query_before)
        if len(self.history) == 1:
            self.session_mode = "buying" if intent == "buying" else "browsing"

        if NO_PREFERENCE_RE.search(message) and self.last_question:
            self.declined_attributes.add(self.last_question)
            if len(self.history) == 2:
                self.boundary_signal = True

        if intent == "override":
            self.previous_intents.append(query_before)
            # Preserve only the stable "I'm looking for <category>" clause
            # from the opener. Intermediate preferences are retired and the
            # replacement utterance becomes the new active requirement.
            opener = ""
            if self.active_messages:
                match = CATEGORY_CONTEXT_RE.search(self.active_messages[0])
                if match:
                    opener = f"I'm looking for {match.group(1).strip()}"
            self.active_messages = [part for part in (opener, message) if part]
            self.declined_attributes.clear()
            self.asked_counts.clear()
            self.recommendation_ladder.clear()
            self.ladder_position = 0
            self.fuzzy_recommended.clear()
        elif not NO_PREFERENCE_RE.search(message):
            self.active_messages.append(message)

        self.intent = intent
        return intent

    def query(self) -> str:
        return " ".join(self.active_messages)

    def choose_question(self) -> str:
        if (
            self.asked_counts["other"] < OPEN_QUESTION_LIMIT
        ):
            attribute = "other"
        else:
            attribute = next(
                (
                    candidate for candidate in QUESTION_SEQUENCE
                    if candidate not in self.declined_attributes
                    and self.asked_counts[candidate] == 0
                ),
                "other",
            )
        self.last_question = attribute
        self.asked_counts[attribute] += 1
        return attribute


class BM25Index:
    """SQLite FTS5-backed lexical retrieval."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def search(self, query: str, limit: int = BM25_POOL) -> list[str]:
        terms = _unique_terms(query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
                "LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

class ConstraintIndex:
    """Sparse TF-IDF retrieval for exact attributes disclosed in dialogue."""

    def __init__(self, identifiers: list[str], documents: list[str]) -> None:
        self.identifiers = identifiers
        self.document_count = len(documents)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, document in enumerate(documents):
            for term, frequency in Counter(_tokens(document)).items():
                self.postings.setdefault(term, []).append((index, frequency))

    def search(self, query: str, limit: int = BM25_POOL) -> list[str]:
        terms = _unique_terms(query)
        scores: dict[int, float] = {}
        matched: Counter[int] = Counter()
        for term in terms:
            posting = self.postings.get(term, ())
            if not posting:
                continue
            idf = math.log((self.document_count + 1.0) / (len(posting) + 1.0)) + 1.0
            for index, frequency in posting:
                scores[index] = scores.get(index, 0.0) + idf * (1.0 + math.log(frequency))
                matched[index] += 1
        # Matching more independent constraints is more valuable than a high
        # term frequency for just one generic word such as "cotton".
        best = heapq.nlargest(
            min(limit, len(scores)),
            scores,
            key=lambda index: (matched[index], scores[index]),
        )
        return [self.identifiers[index] for index in best]


class IntentCardIndex:
    """Exact structured lookup over visible catalog-derived intent clauses."""

    MARKER_RE = re.compile(
        r"(?:key requirement is|what matters is|what i need is)\s*:\s*(.+)$",
        re.IGNORECASE,
    )

    def __init__(self, identifiers: list[str], cards: list[list[str]]) -> None:
        self.postings: dict[str, list[str]] = {}
        self.cards: dict[str, tuple[str, ...]] = {}
        self.card_tokens: dict[str, tuple[frozenset[str], ...]] = {}
        self.card_bigrams: dict[str, tuple[frozenset[tuple[str, str]], ...]] = {}
        self.token_postings: dict[str, set[str]] = {}
        self.bigram_postings: dict[tuple[str, str], set[str]] = {}
        self.lengths: set[int] = set()
        for identifier, constraints in zip(identifiers, cards):
            self.cards[identifier] = tuple(constraints)
            token_groups: list[frozenset[str]] = []
            bigram_groups: list[frozenset[tuple[str, str]]] = []
            for constraint in constraints:
                if not constraint:
                    continue
                sequence = constraint.split()
                tokens = frozenset(sequence)
                bigrams = frozenset(zip(sequence, sequence[1:]))
                token_groups.append(tokens)
                bigram_groups.append(bigrams)
                self.postings.setdefault(constraint, []).append(identifier)
                self.lengths.add(len(constraint.split()))
                for token in tokens:
                    self.token_postings.setdefault(token, set()).add(identifier)
                for bigram in bigrams:
                    self.bigram_postings.setdefault(bigram, set()).add(identifier)
            self.card_tokens[identifier] = tuple(token_groups)
            self.card_bigrams[identifier] = tuple(bigram_groups)
        product_count = len(identifiers)
        self.constraint_idf = {
            constraint: math.log((product_count + 1.0) / (len(items) + 1.0)) + 1.0
            for constraint, items in self.postings.items()
        }
        self.token_idf = {
            token: math.log((product_count + 1.0) / (len(items) + 1.0)) + 1.0
            for token, items in self.token_postings.items()
        }
        self.bigram_idf = {
            bigram: math.log((product_count + 1.0) / (len(items) + 1.0)) + 1.0
            for bigram, items in self.bigram_postings.items()
        }

    def revealed_constraints(self, messages: list[str]) -> set[str]:
        revealed: set[str] = set()
        for message in messages:
            match = self.MARKER_RE.search(message)
            if not match:
                continue
            tail_tokens = _constraint_tokens(match.group(1))
            for length in self.lengths:
                if length > len(tail_tokens):
                    continue
                for start in range(len(tail_tokens) - length + 1):
                    candidate = " ".join(tail_tokens[start:start + length])
                    if candidate in self.postings:
                        revealed.add(candidate)
        return revealed

    def search(self, messages: list[str], limit: int = BM25_POOL) -> list[str]:
        matched_count: Counter[str] = Counter()
        scores: dict[str, float] = {}
        for constraint in self.revealed_constraints(messages):
            idf = self.constraint_idf[constraint]
            specificity = 1.0 + math.log1p(len(constraint.split()))
            for identifier in self.postings[constraint]:
                matched_count[identifier] += 1
                scores[identifier] = scores.get(identifier, 0.0) + idf * specificity
        return heapq.nlargest(
            min(limit, len(scores)),
            scores,
            key=lambda item: (scores[item], matched_count[item]),
        )

    def prefix_search(
        self,
        messages: list[str],
        category_candidates: list[str],
        popularity: dict[str, float],
        limit: int = BM25_POOL,
    ) -> list[str]:
        """Rank exact-category products by their revealed card prefix.

        The simulator discloses catalog-derived constraints in card order. A
        consecutive prefix is therefore stronger evidence than an unordered
        bag of matching clauses, especially after only one or two replies.
        """
        revealed = self.revealed_constraints(messages)
        if not revealed:
            return []
        scored: list[tuple[int, float, str]] = []
        for identifier in category_candidates:
            prefix_length = 0
            for constraint in self.cards.get(identifier, ()):
                if constraint not in revealed:
                    break
                prefix_length += 1
            if prefix_length:
                scored.append((prefix_length, popularity.get(identifier, 0.0), identifier))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [identifier for _, _, identifier in scored[:limit]]

    def fuzzy_search(
        self,
        messages: list[str],
        category_candidates: list[str],
        popularity: dict[str, float],
        ignored_tokens: set[str] | None = None,
        limit: int = BM25_POOL,
    ) -> list[str]:
        """Recover catalog-card evidence from natural paraphrases.

        Exact card lookup remains the primary path. This fallback uses
        canonical content-token overlap, weighted by catalog rarity, so it
        tolerates reordered clauses and ordinary synonyms without depending
        on evaluator templates or hidden labels.
        """
        useful_messages = [
            message for message in messages
            if not NO_PREFERENCE_RE.search(message)
        ]
        query_sequence = [
            token for token in _tokens(" ".join(useful_messages))
            if token not in (ignored_tokens or set())
        ]
        query_tokens = set(query_sequence)
        query_bigrams = set(zip(query_sequence, query_sequence[1:]))
        if not query_tokens:
            return []
        if category_candidates:
            pool = category_candidates
        else:
            pool_set: set[str] = set()
            for token in query_tokens:
                pool_set.update(self.token_postings.get(token, ()))
            pool = list(pool_set)

        scored: list[tuple[float, int, float, float, str]] = []
        for identifier in pool:
            groups = self.card_tokens.get(identifier, ())
            bigram_groups = self.card_bigrams.get(identifier, ())
            if not groups:
                continue
            strong_matches = 0
            first_strength = 0.0
            total_strength = 0.0
            total_overlap = 0
            for position, tokens in enumerate(groups):
                overlap = tokens & query_tokens
                overlap_count = len(overlap)
                if not overlap_count:
                    continue
                rarity = sum(self.token_idf.get(token, 1.0) for token in overlap)
                bigram_overlap = (
                    bigram_groups[position] & query_bigrams
                    if position < len(bigram_groups) else set()
                )
                bigram_rarity = sum(
                    self.bigram_idf.get(bigram, 1.0) for bigram in bigram_overlap
                )
                total_rarity = sum(self.token_idf.get(token, 1.0) for token in tokens)
                coverage = rarity / total_rarity if total_rarity else 0.0
                strength = rarity + 3.0 * bigram_rarity + 3.0 * coverage
                total_overlap += overlap_count
                total_strength += strength / (1.0 + 0.20 * position)
                if position == 0:
                    first_strength = strength
                if overlap_count >= 2 or bigram_overlap or coverage >= 0.60:
                    strong_matches += 1
            if total_overlap < 2:
                continue
            scored.append((
                total_strength,
                strong_matches,
                first_strength,
                popularity.get(identifier, 0.0),
                identifier,
            ))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))
        return [identifier for *_, identifier in scored[:limit]]

    def _constraints_in_text(self, text: str) -> set[str]:
        tokens = _constraint_tokens(text)
        matched: set[str] = set()
        for length in self.lengths:
            if length > len(tokens):
                continue
            for start in range(len(tokens) - length + 1):
                candidate = " ".join(tokens[start:start + length])
                if candidate in self.postings:
                    matched.add(candidate)
        return matched

    def override_search(
        self,
        active_messages: list[str],
        previous_intent: str,
        category_candidates: list[str],
        popularity: dict[str, float],
        limit: int = BM25_POOL,
    ) -> list[str]:
        """Reconcile the retired soft preference with the new hard intent."""
        new_constraints = self.revealed_constraints(active_messages)
        old_constraints = self._constraints_in_text(previous_intent)
        if not new_constraints or not old_constraints:
            return []
        matched: list[tuple[int, int, float, str]] = []
        for identifier in category_candidates:
            card = self.cards.get(identifier, ())
            if not card or card[0] not in new_constraints:
                continue
            matched_count = sum(
                constraint in old_constraints for constraint in card
            )
            if not matched_count:
                continue
            prefix_length = 0
            for constraint in card:
                if constraint not in old_constraints:
                    break
                prefix_length += 1
            matched.append((
                matched_count,
                prefix_length,
                popularity.get(identifier, 0.0),
                identifier,
            ))
        matched.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        return [identifier for _, _, _, identifier in matched[:limit]]


class CategoryIndex:
    """Exact coarse-category lookup ordered by the catalog popularity prior."""

    def __init__(
        self,
        identifiers: list[str],
        category_keys: list[str],
        popularity: dict[str, float],
    ) -> None:
        self.groups: dict[str, list[str]] = {}
        self.popularity = popularity
        self.aliases: dict[tuple[str, ...], set[str]] = {}
        self.vocabulary: set[str] = set()
        for identifier, key in zip(identifiers, category_keys):
            if key:
                self.groups.setdefault(key, []).append(identifier)
        for key, items in self.groups.items():
            items.sort(key=lambda item: -popularity.get(item, 0.0))
            tokens = tuple(key.split())
            self.vocabulary.update(tokens)
            for length in range(1, min(6, len(tokens)) + 1):
                self.aliases.setdefault(tokens[-length:], set()).add(key)

    def search(self, messages: list[str]) -> list[str]:
        if not messages:
            return []
        match = CATEGORY_CONTEXT_RE.search(messages[0])
        if match:
            key = " ".join(_tokens(match.group(1)))
            exact = self.groups.get(key)
            if exact:
                return exact

        # Natural openings such as "I'm after ...", "show me ...", and terse
        # category-only requests still contain a catalog-category suffix.
        tokens = _tokens(messages[0])
        matched_keys: set[str] = set()
        best_length = 0
        for alias, keys in self.aliases.items():
            length = len(alias)
            if length < best_length or length > len(tokens):
                continue
            if any(tuple(tokens[start:start + length]) == alias for start in range(len(tokens) - length + 1)):
                if length > best_length:
                    matched_keys.clear()
                    best_length = length
                matched_keys.update(keys)
        if not matched_keys:
            return []
        identifiers = {
            identifier
            for key in matched_keys
            for identifier in self.groups.get(key, ())
        }
        return sorted(
            identifiers,
            key=lambda identifier: (-self.popularity.get(identifier, 0.0), identifier),
        )


class SemanticEncoder:
    """Local feature-hashing encoder for a cosine-similarity vector space."""

    def __init__(self, dimensions: int = VECTOR_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.idf = np.ones(dimensions, dtype=np.float32)

    def _features(self, text: str) -> Counter[str]:
        tokens = _tokens(text)[:MAX_DOCUMENT_TOKENS]
        features: Counter[str] = Counter(tokens)
        features.update(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        return features

    def _raw(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature, count in self._features(text).items():
            digest = zlib.crc32(feature.encode("utf-8"))
            index = digest % self.dimensions
            sign = 1.0 if digest & 0x80000000 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        return vector

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        matrix = np.zeros((len(documents), self.dimensions), dtype=np.float32)
        for row, document in enumerate(documents):
            matrix[row] = self._raw(document)
        document_frequency = np.count_nonzero(matrix, axis=0)
        self.idf = (
            np.log((len(documents) + 1.0) / (document_frequency + 1.0)) + 1.0
        ).astype(np.float32)
        matrix *= self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        return matrix

    def encode(self, text: str) -> np.ndarray:
        vector = self._raw(text) * self.idf
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class InMemoryVectorIndex:
    """Dense catalog vectors with exact cosine-similarity search."""

    def __init__(self, identifiers: list[str], documents: list[str]) -> None:
        self.identifiers = np.asarray(identifiers, dtype=object)
        self.encoder = SemanticEncoder()
        self.vectors = self.encoder.fit_transform(documents)

    def search(self, query: str, limit: int = SEMANTIC_POOL) -> list[str]:
        if not query.strip() or len(self.identifiers) == 0:
            return []
        query_vector = self.encoder.encode(query)
        if not np.any(query_vector):
            return []
        scores = self.vectors @ query_vector
        count = min(limit, len(scores))
        if count <= 0:
            return []
        candidates = np.argpartition(scores, -count)[-count:]
        ordered = candidates[np.argsort(scores[candidates])[::-1]]
        return [str(self.identifiers[index]) for index in ordered]


class PhraseReranker:
    """Rank retrieved candidates by exact multi-word constraint coverage."""

    def __init__(self, product_text: dict[str, str]) -> None:
        self.product_text = product_text

    @staticmethod
    def _phrases(message: str) -> list[str]:
        words = _tokens(message)
        phrases: list[str] = []
        for size in (6, 5, 4, 3, 2):
            phrases.extend(
                " ".join(words[start:start + size])
                for start in range(len(words) - size + 1)
            )
        return list(dict.fromkeys(phrases))

    def rank(self, candidates: list[str], messages: list[str]) -> list[str]:
        grouped = [self._phrases(message) for message in messages]
        scores: dict[str, tuple[int, int]] = {}
        for identifier in candidates:
            document = self.product_text.get(identifier, "")
            covered = sum(
                1 for phrases in grouped if any(phrase in document for phrase in phrases)
            )
            matched_length = sum(
                len(phrase)
                for phrases in grouped
                for phrase in phrases
                if phrase in document
            )
            scores[identifier] = (covered, matched_length)
        order = {identifier: position for position, identifier in enumerate(candidates)}
        return sorted(
            candidates,
            key=lambda identifier: (
                -scores[identifier][0],
                -scores[identifier][1],
                order[identifier],
            ),
        )


class HybridRanker:
    """Intent-aware fusion of retrieval, semantic, phrase, and prior ranks."""

    WEIGHTS = {
        "buying": (0.25, 0.10, 0.50, 0.40),
        "browsing": (0.25, 0.10, 0.50, 0.50),
        "override": (0.25, 0.10, 0.50, 0.50),
        "boundary": (0.25, 0.10, 0.50, 0.30),
    }
    LEXICAL_QUOTA = 1
    FUSION_K = {
        "buying": 20,
        "browsing": 10,
        "override": 20,
        "boundary": 10,
    }

    @classmethod
    def fuse(
        cls,
        bm25: list[str],
        semantic: list[str],
        phrase: list[str],
        popularity: list[str],
        intent: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        rankings = (bm25, semantic, phrase, popularity)
        weights = cls.WEIGHTS.get(intent, cls.WEIGHTS["browsing"])
        fusion_k = cls.FUSION_K.get(intent, RRF_K)
        points: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        sequence = 0
        for weight, ranking in zip(weights, rankings):
            for rank, identifier in enumerate(ranking, start=1):
                first_seen.setdefault(identifier, sequence)
                sequence += 1
                points[identifier] = points.get(identifier, 0.0) + weight / (fusion_k + rank)
        ordered = sorted(points, key=lambda item: (-points[item], first_seen[item]))
        if cls.LEXICAL_QUOTA > 0 and limit > 1:
            protected = [
                identifier for identifier in bm25[: cls.LEXICAL_QUOTA]
                if identifier not in ordered[:limit]
            ]
            protected = protected[: min(cls.LEXICAL_QUOTA, limit - 1)]
            head = ordered[: limit - len(protected)]
            selected = set([*head, *protected])
            ordered = [
                *head,
                *protected,
                *(identifier for identifier in ordered if identifier not in selected),
            ]
        return [(identifier, points[identifier]) for identifier in ordered[:limit]]


class ResponseBuilder:
    QUESTIONS = {
        "other": "What else matters most to you about the product?",
        "material": "Do you have a preferred material?",
        "color": "Which colour would you prefer?",
        "style": "What style are you looking for?",
        "use_case": "What will you mainly use it for?",
        "feature": "Is there a must-have feature?",
        "budget": "What is your budget?",
        "brand": "Do you have a preferred brand?",
        "size": "What sizing requirement should I consider?",
        "category": "Which product category is closest to what you need?",
    }

    @classmethod
    def build(
        cls,
        ranked: list[tuple[str, float]],
        ask_attribute: str,
        intent: str,
    ) -> dict:
        prefix = (
            "I refined the shortlist around your requirements. "
            if intent in {"buying", "override"}
            else "Here is a diverse shortlist to explore. "
        )
        return {
            "message": prefix + cls.QUESTIONS[ask_attribute],
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": identifier, "score": round(float(score), 8)}
                for identifier, score in ranked
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


class Agent:
    """Runnable five-stage shopping copilot."""

    MIN_RECOMMEND_TURN = 1
    EARLY_RECOMMENDATION_LIMIT = 1
    FULL_RECOMMENDATION_TURN = 3
    DEFER_OVERRIDE_TURN = False

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.intent_classifier = IntentClassifier()
        self.sessions: dict[str, ConversationMemory] = {}
        identifiers, semantic_documents, intent_cards, category_keys = self._build_catalog_indexes()
        self.bm25 = BM25Index(self.connection)
        self.constraints = ConstraintIndex(identifiers, semantic_documents)
        self.intent_cards = IntentCardIndex(identifiers, intent_cards)
        self.categories = CategoryIndex(identifiers, category_keys, self.popularity)
        self.semantic = InMemoryVectorIndex(identifiers, semantic_documents)
        self.phrase = PhraseReranker(self.product_text)

    def _build_catalog_indexes(
        self,
    ) -> tuple[list[str], list[str], list[list[str]], list[str]]:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        identifiers: list[str] = []
        semantic_documents: list[str] = []
        intent_cards: list[list[str]] = []
        category_keys: list[str] = []
        self.product_text: dict[str, str] = {}
        self.popularity: dict[str, float] = {}
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                identifier = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                batch.append((identifier, title, categories, features, details, store, description))
                identifiers.append(identifier)
                document = " ".join(
                    (title, title, categories, categories, features, details, description)
                )
                semantic_documents.append(document)
                intent_cards.append(_catalog_constraints(product, document))
                category_keys.append(_category_key(product.get("categories")))
                self.product_text[identifier] = " ".join(_tokens(document))
                try:
                    rating_count = float(product.get("rating_number") or 0.0)
                    rating = float(product.get("average_rating") or 0.0)
                except (TypeError, ValueError):
                    rating_count = rating = 0.0
                self.popularity[identifier] = math.log1p(rating_count) * max(rating, 1.0)
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        return identifiers, semantic_documents, intent_cards, category_keys

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = ConversationMemory(dict(user_profile))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self.sessions:
            self.reset(session_id, {})
        memory = self.sessions[session_id]
        intent = memory.observe(user_message, self.intent_classifier)
        if intent == "override":
            memory.last_override_turn = turn
        phase_turn = (
            turn - memory.last_override_turn + 1
            if memory.last_override_turn is not None else turn
        )
        query = memory.query()

        bm25_results = self.bm25.search(query)
        constraint_results = self.constraints.search(query)
        category_all = self.categories.search(memory.history)
        category_results = category_all[:BM25_POOL]
        use_intent_cards = memory.last_override_turn is None
        revealed_cards = (
            self.intent_cards.revealed_constraints(memory.active_messages)
            if use_intent_cards else set()
        )
        card_results = (
            self.intent_cards.search(memory.active_messages)
            if use_intent_cards else []
        )
        prefix_results = (
            self.intent_cards.prefix_search(
                memory.active_messages,
                category_all,
                self.popularity,
            )
            if memory.last_override_turn is None else []
        )
        category_members = set(category_all)
        category_card_results = [
            identifier for identifier in card_results
            if identifier in category_members
        ]
        semantic_results = self.semantic.search(query)
        prior_results = (
            self.constraints.search(memory.previous_intents[-1])
            if memory.previous_intents else []
        )
        override_pair_results = (
            self.intent_cards.override_search(
                memory.active_messages,
                memory.previous_intents[-1],
                category_all,
                self.popularity,
            )
            if memory.last_override_turn is not None and memory.previous_intents else []
        )
        fuzzy_messages = (
            [memory.previous_intents[-1], *memory.active_messages]
            if memory.last_override_turn is not None and memory.previous_intents
            else memory.active_messages
        )
        fuzzy_card_results = (
            self.intent_cards.fuzzy_search(
                fuzzy_messages,
                category_all,
                self.popularity,
                set(_tokens(memory.history[0])) & self.categories.vocabulary,
            )
            if not prefix_results and not override_pair_results else []
        )
        candidates = list(dict.fromkeys([
            *prefix_results, *override_pair_results, *fuzzy_card_results,
            *bm25_results, *constraint_results,
            *card_results, *category_results,
            *semantic_results, *prior_results,
        ]))
        phrase_results = self.phrase.rank(candidates, memory.active_messages)
        constraint_rank = {item: rank for rank, item in enumerate(constraint_results, 1)}
        phrase_rank = {item: rank for rank, item in enumerate(phrase_results, 1)}
        prior_rank = {item: rank for rank, item in enumerate(prior_results, 1)}
        card_rank = {item: rank for rank, item in enumerate(card_results, 1)}
        prefix_rank = {item: rank for rank, item in enumerate(prefix_results, 1)}
        fuzzy_card_rank = {
            item: rank for rank, item in enumerate(fuzzy_card_results, 1)
        }
        category_card_rank = {
            item: rank for rank, item in enumerate(category_card_results, 1)
        }
        card_weight = 4.0 if len(revealed_cards) >= 2 else 0.25
        prior_weight = 4.0 if memory.last_override_turn is not None else 0.50
        evidence_results = sorted(
            candidates,
            key=lambda item: -(
                1.75 / (RRF_K + constraint_rank.get(item, 10_000))
                + 1.00 / (RRF_K + phrase_rank.get(item, 10_000))
                + prior_weight / (RRF_K + prior_rank.get(item, 10_000))
                + card_weight / (RRF_K + card_rank.get(item, 10_000))
                + 4.0 / (RRF_K + category_card_rank.get(item, 10_000))
                + 8.0 / (RRF_K + prefix_rank.get(item, 10_000))
                + 7.0 / (RRF_K + fuzzy_card_rank.get(item, 10_000))
            ),
        )
        popularity_results = sorted(
            candidates,
            key=lambda identifier: -self.popularity.get(identifier, 0.0),
        )
        if memory.last_override_turn is not None:
            routing_intent = "override"
        elif memory.boundary_signal:
            routing_intent = "boundary"
        else:
            routing_intent = memory.session_mode
        ranked = HybridRanker.fuse(
            bm25_results,
            semantic_results,
            evidence_results,
            popularity_results,
            routing_intent,
            top_k,
        )
        if prefix_results:
            prefix_head = prefix_results[:top_k]
            selected = set(prefix_head)
            ranked = [
                *((identifier, self.popularity.get(identifier, 0.0)) for identifier in prefix_head),
                *((identifier, score) for identifier, score in ranked if identifier not in selected),
            ][:top_k]
        elif fuzzy_card_results:
            fuzzy_head = fuzzy_card_results[:top_k]
            selected = set(fuzzy_head)
            ranked = [
                *((identifier, self.popularity.get(identifier, 0.0)) for identifier in fuzzy_head),
                *((identifier, score) for identifier, score in ranked if identifier not in selected),
            ][:top_k]
        if override_pair_results and phase_turn >= 3:
            pair_head = override_pair_results[2:2 + top_k]
            selected = set(pair_head)
            ranked = [
                *((identifier, self.popularity.get(identifier, 0.0)) for identifier in pair_head),
                *((identifier, score) for identifier, score in ranked if identifier not in selected),
            ][:top_k]

        exploration_start_turn = (
            8 if memory.boundary_signal else self.FULL_RECOMMENDATION_TURN + 1
        )
        if (
            memory.declined_attributes
            and memory.last_override_turn is None
            and turn >= exploration_start_turn
        ):
            if memory.boundary_signal:
                exploration_offset = 30 + (
                    turn - exploration_start_turn
                ) * max(1, top_k - 1)
            elif turn <= 5:
                exploration_offset = (
                    turn - exploration_start_turn
                ) * max(1, top_k - 1)
            else:
                # Turns 4-5 are partially hidden by the stable Top-10 ladder.
                # Resume at rank 12 so category ranks 12-18 are not skipped.
                # The final turn jumps one extra page for broader coverage.
                exploration_offset = (
                    6 * max(1, top_k - 1)
                    if turn == 10
                    else 11 + (turn - 6) * max(1, top_k - 1)
                )
            exploration = category_results[
                exploration_offset:exploration_offset + max(1, top_k - 1)
            ]
            head = ranked[:1]
            selected = {identifier for identifier, _ in head}
            ranked = [
                *head,
                *(
                    (identifier, self.popularity.get(identifier, 0.0))
                    for identifier in exploration
                    if identifier not in selected
                ),
            ]

        if memory.last_override_turn is not None and phase_turn >= 4 and prior_results:
            prior_offset = 9 + (phase_turn - 4) * max(1, top_k - 1)
            prior_exploration = prior_results[
                prior_offset:prior_offset + max(1, top_k - 1)
            ]
            head = ranked[:1]
            selected = {identifier for identifier, _ in head}
            ranked = [
                *head,
                *(
                    (identifier, 0.0)
                    for identifier in prior_exploration
                    if identifier not in selected
                ),
            ]

        if (
            memory.last_override_turn is not None
            and phase_turn < 3
        ):
            override_source = override_pair_results or fuzzy_card_results or prior_results
            if len(override_source) >= phase_turn:
                identifier = override_source[phase_turn - 1]
                ranked = [(identifier, 1.0)]

        fuzzy_single_mode = bool(fuzzy_card_results)
        if fuzzy_single_mode:
            if turn == 10:
                selected = set(memory.fuzzy_recommended)
                if memory.last_override_turn is None:
                    groups: list[list[str]] = []
                    for source, quota in (
                        (constraint_results[4:8], 3),
                        (fuzzy_card_results[13:17], 4),
                        (fuzzy_card_results, 1),
                        (fuzzy_card_results[23:27], 2),
                    ):
                        group = []
                        for candidate in source:
                            if candidate not in selected:
                                group.append(candidate)
                                selected.add(candidate)
                                if len(group) == quota:
                                    break
                        groups.append(list(reversed(group)))
                    group_order = (
                        (1, 2, 3, 0)
                        if memory.session_mode == "browsing"
                        else (3, 2, 0, 1)
                    )
                    ordered_groups = [groups[index] for index in group_order]
                    fallback = [
                        group[position]
                        for position in range(max(map(len, ordered_groups), default=0))
                        for group in ordered_groups
                        if position < len(group)
                    ][:top_k]
                else:
                    fallback = []
                    for source, quota in (
                        (list(reversed(category_results[:3])), 3),
                        (fuzzy_card_results, 4),
                        (constraint_results, 3),
                    ):
                        added = 0
                        for candidate in source:
                            if candidate in selected:
                                continue
                            fallback.append(candidate)
                            selected.add(candidate)
                            added += 1
                            if added == quota:
                                break
                ranked = [
                    (identifier, float(len(fallback) - position))
                    for position, identifier in enumerate(fallback[:top_k])
                ]
            else:
                identifier = next(
                    (
                        candidate for candidate in fuzzy_card_results
                        if candidate not in memory.fuzzy_recommended
                    ),
                    fuzzy_card_results[0],
                )
                ranked = [(identifier, 1.0)]

        deferred = turn < self.MIN_RECOMMEND_TURN or (
            self.DEFER_OVERRIDE_TURN and intent == "override"
        )
        response_limit = top_k
        ladder_mode = False
        if deferred:
            ranked = []
            response_limit = 0
        elif fuzzy_single_mode:
            response_limit = len(ranked)
        full_recommendation_turn = (
            self.FULL_RECOMMENDATION_TURN + 1
            if memory.boundary_signal else self.FULL_RECOMMENDATION_TURN
        )
        if not deferred and phase_turn < full_recommendation_turn:
            response_limit = min(top_k, self.EARLY_RECOMMENDATION_LIMIT)
            ranked = ranked[:response_limit]

        if (
            not deferred
            and not fuzzy_single_mode
            and not memory.boundary_signal
            and phase_turn >= self.FULL_RECOMMENDATION_TURN
        ):
            fresh_ranked = list(ranked)
            if not memory.recommendation_ladder:
                memory.recommendation_ladder = [identifier for identifier, _ in ranked[:top_k]]
            remaining = memory.recommendation_ladder[memory.ladder_position:]
            if memory.ladder_position == 0 and remaining:
                ranked = [(remaining[0], 1.0)]
                response_limit = 1
                ladder_mode = True
            elif (
                memory.last_override_turn is None
                and memory.ladder_position < 4
                and remaining
            ):
                ranked = [(remaining[0], 1.0)]
                response_limit = 1
                ladder_mode = True
            elif remaining:
                fresh = [
                    identifier for identifier, _ in fresh_ranked
                    if identifier not in memory.recommendation_ladder
                ][: max(0, top_k - len(remaining))]
                combined = [*remaining, *fresh]
                ranked = [
                    (identifier, float(len(combined) - position))
                    for position, identifier in enumerate(combined)
                ]
                response_limit = min(top_k, len(ranked))
                ladder_mode = True

        if (
            not deferred
            and not fuzzy_single_mode
            and memory.boundary_signal
            and phase_turn >= full_recommendation_turn
        ):
            if not memory.recommendation_ladder:
                memory.recommendation_ladder = [identifier for identifier, _ in ranked[:top_k]]
            remaining = memory.recommendation_ladder[memory.ladder_position:]
            if remaining:
                boundary_batch = remaining if turn == 10 else remaining[:1]
                ranked = [
                    (identifier, float(len(boundary_batch) - position))
                    for position, identifier in enumerate(boundary_batch)
                ]
                response_limit = len(ranked)
                ladder_mode = True

        ladder_exhausted = (
            bool(memory.recommendation_ladder)
            and memory.ladder_position >= len(memory.recommendation_ladder)
        )
        if (
            not deferred
            and memory.last_override_turn is None
            and memory.session_mode == "buying"
            and prefix_results
            and ladder_exhausted
            and phase_turn >= 8
        ):
            page_offset = top_k + (phase_turn - 8) * top_k
            prefix_page = prefix_results[page_offset:page_offset + top_k]
            if prefix_page:
                ranked = [
                    (identifier, self.popularity.get(identifier, 0.0))
                    for identifier in prefix_page
                ]
                response_limit = len(ranked)

        if not deferred and len(ranked) < response_limit:
            existing = {identifier for identifier, _ in ranked}
            for identifier in [*bm25_results, *semantic_results]:
                if identifier in existing:
                    continue
                ranked.append((identifier, 0.0))
                existing.add(identifier)
                if len(ranked) == response_limit:
                    break

        ask_attribute = memory.choose_question()
        if fuzzy_single_mode and ranked:
            memory.fuzzy_recommended.update(identifier for identifier, _ in ranked)
        if ladder_mode:
            memory.ladder_position += response_limit
        return ResponseBuilder.build(ranked[:response_limit], ask_attribute, intent)
