"""Shared text-normalisation helpers, compiled regexes, and tuning constants.

Leaf module: depends only on the standard library. Imported by the memory,
retrieval, ranking, and agent modules. No behaviour lives here beyond pure
functions and constants — moving code out of the former monolithic
``agent.py`` must not change any of it.
"""

from __future__ import annotations

import re

__all__ = [
    "TOKEN_RE", "CONSTRAINT_TOKEN_RE", "CATEGORY_CONTEXT_RE", "OVERRIDE_RE",
    "MATERIAL_VALUE_RE", "COLOR_VALUE_RE",
    "STOPWORDS", "CONCEPT_ALIASES", "PHRASE_ALIASES", "QUESTION_SEQUENCE",
    "BM25_POOL", "SEMANTIC_POOL", "VECTOR_DIMENSIONS", "RRF_K",
    "MAX_QUERY_TERMS", "MAX_DOCUMENT_TOKENS", "OPEN_QUESTION_LIMIT",
    "RERANK_WINDOW",
    "_text", "_canonical_text", "_tokens", "_constraint_tokens",
    "_unique_terms", "_flatten_values", "_catalog_constraints",
    "_category_key",
]


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CONSTRAINT_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
CATEGORY_CONTEXT_RE = re.compile(
    r"^\s*i['’]?m\s+looking\s+for\s+([^.,;]+)", re.IGNORECASE
)
# Retraction / topic-change cues that mark a genuine intent override (the
# shopper abandoning a previously stated preference). These are natural-language
# reset phrases — retraction verbs ("scratch/forget/ignore/scrap"), reversals
# ("on second thought", "changed my mind", "never mind"), and substitutions
# ("instead of", "rather than", "switch to") — none of which occur in the
# simulators' ordinary constraint replies, so this does not fire on normal
# turns. Detection is additionally gated to non-opening turns (see
# ``IntentClassifier.classify``), so a first-message phrasing never trips it.
OVERRIDE_RE = re.compile(
    r"\b(?:"
    r"scratch\s+(?:the|that|my)|"
    r"scrap\s+(?:the|that|my)|"
    r"forget\s+(?:the|that|my|about|what)|"
    r"ignore\s+(?:the|that|my|earlier|previous|prior|what)|"
    r"disregard\s+(?:the|that|my|earlier|previous|prior|what)|"
    r"never\s*mind|"
    r"on\s+second\s+thought|"
    r"changed?\s+my\s+mind|"
    r"instead\s+of|"
    r"rather\s+than|"
    r"switch(?:ing)?\s+(?:to|gears)"
    r")\b",
    re.IGNORECASE,
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
RERANK_WINDOW = 30

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
