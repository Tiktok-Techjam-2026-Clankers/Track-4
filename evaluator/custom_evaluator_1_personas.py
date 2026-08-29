"""Deterministic customer personas for custom evaluator 1.

A persona controls only how the simulated customer *words* things. Which
constraints are revealed, and on which turn, stays under the official simulator
policy in `evaluator.local_evaluator`, so any metric difference between personas
is attributable to phrasing alone.

The `verbatim` persona reproduces the official wording exactly and acts as the
calibration control.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from evaluator.local_evaluator import COLOR_RE, MATERIALS


STOPWORDS = frozenset(
    "a an the and or but of for with without to from in on at by as is are be been am "
    "this that these those it its your our their you we they i my me will can could "
    "would should has have had do does did so very really just about into over than "
    "there here what which who when where how all any both each more most other some "
    "such only own same too s t"
    .split()
)

KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /&'’-]{0,32}?)\s*:\s*(.+?)\s*$")
PARENTHETICAL_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[;,]|\s[-–—]\s|\|)\s*")
NON_WORD_RE = re.compile(r"[^\w$%.\- ]+")
POSSESSIVE_RE = re.compile(r"\b(women|men|girl|boy|kid|baby|babies)(?:'|’)?s\b", re.I)

DEPARTMENTS = {
    "womens": "women", "women": "women", "mens": "men", "men": "men",
    "girls": "girls", "boys": "boys", "baby": "babies", "kids": "kids",
    "unisex-adult": "adults", "unisex adult": "adults", "unisex": "anyone",
    "unisex-child": "kids", "unisex-baby": "babies",
}

# Detail keys arrive as "Key: value" from the official intent card. Rewriting them
# into natural clauses removes the catalogue's key-value shape while keeping the
# value word intact, so the constraint stays satisfiable.
DETAIL_PHRASES = {
    "department": "made for {value}",
    "color": "in {value}",
    "colour": "in {value}",
    "material": "made of {value}",
    "material composition": "made of {value}",
    "fabric type": "made of {value}",
    "outer material": "made of {value}",
    "inner material": "lined with {value}",
    "sole material": "with a {value} sole",
    "closure type": "with a {value} closure",
    "sleeve type": "with {value} sleeves",
    "neck style": "with a {value} neckline",
    "fit type": "cut {value}",
    "style": "in a {value} style",
    "size": "in size {value}",
    "brand": "from {value}",
    "manufacturer": "from {value}",
    "care instructions": "and it should be {value}",
    "heel type": "with a {value} heel",
    "heel height": "with a {value} heel",
    "shaft height": "with a {value} shaft",
    "water resistance level": "rated {value} for water",
    "season": "for {value}",
    "occasion": "for {value}",
    "pattern": "with a {value} pattern",
    "theme": "on a {value} theme",
    "age range description": "for {value}",
}

# Meaning-preserving but lexically disjoint from the catalogue text. A customer who
# says "airy" about a product the catalogue calls "breathable" is the realistic
# failure mode this suite is built to measure.
SYNONYMS = {
    "lightweight": "light",
    "breathable": "airy",
    "moisture wicking": "sweat wicking",
    "moisture-wicking": "sweat wicking",
    "sweat absorbent": "soaks up sweat",
    "quick dry": "dries fast",
    "quick-dry": "dries fast",
    "comfortable": "comfy",
    "durable": "hard wearing",
    "sturdy": "solid",
    "machine washable": "washes in the machine",
    "hand wash": "washed by hand",
    "long sleeve": "long sleeved",
    "short sleeve": "short sleeved",
    "high waisted": "high rise",
    "high-waisted": "high rise",
    "elastic waistband": "stretchy waist",
    "drawstring": "pull cord",
    "slip resistant": "non slip",
    "slip-resistant": "non slip",
    "stretchy": "with some give",
    "wrinkle resistant": "does not crease",
    "hypoallergenic": "gentle on skin",
    "adjustable": "easy to adjust",
    "casual": "everyday",
    "lined": "with a lining",
    "insulated": "keeps heat in",
    "warm": "cosy",
    "waterproof": "keeps water out",
}

SYNONYM_RE = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in sorted(SYNONYMS, key=len, reverse=True)) + r")\b",
    re.I,
)

OPENING_TEMPLATES = {
    "buying": {
        "paraphrase": (
            "Hi, I'm after {category}, and it needs to be {constraint}.",
            "I want to buy {category}. The thing I really care about: {constraint}.",
            "Looking for {category}, and {constraint} is a must for me.",
            "I need {category}. Main thing is {constraint}.",
        ),
        "terse": (
            "{category}, {constraint}",
            "need {category} {constraint}",
            "{constraint} {category}?",
        ),
    },
    "browsing": {
        "paraphrase": (
            "Not sure exactly what I want yet, something along the lines of {category}?",
            "Just browsing {category} for now, nothing fixed in mind.",
            "I'm in the market for {category} but I haven't narrowed it down.",
            "Show me some {category}, I'm still deciding.",
        ),
        "terse": (
            "{category}, just browsing",
            "{category}?",
            "looking at {category}",
        ),
    },
    "override": {
        "paraphrase": (
            "I'm after {category}. {constraint}",
            "Shopping for {category}, {constraint}",
            "I need {category}, and {constraint}",
        ),
        "terse": (
            "{category}, {constraint}",
            "{category} {constraint}",
        ),
    },
}

REVEAL_TEMPLATES = {
    "paraphrase": (
        "For that, {joined}.",
        "What matters there is {joined}.",
        "On that front: {joined}.",
        "Yeah, {joined}.",
    ),
    "terse": (
        "{joined}",
        "{joined}.",
    ),
}

NO_PREFERENCE_TEMPLATES = {
    "paraphrase": (
        "No strong feelings on {attribute}.",
        "{attribute} doesn't really matter to me.",
        "Nothing to add on {attribute}.",
    ),
    "terse": (
        "no preference on {attribute}",
        "{attribute}: whatever",
    ),
}

BOUNDARY_TEMPLATES = {
    "paraphrase": (
        "No preference on {attribute}, your call.",
        "I'll leave {attribute} up to you.",
        "Honestly I don't mind about {attribute}, pick for me.",
    ),
    "terse": (
        "{attribute}: up to you",
        "no preference, you pick",
    ),
}

NUDGE_TEMPLATES = {
    "paraphrase": (
        "Not quite what I had in mind. Ask me about one specific thing?",
        "Still not right. What do you need to know?",
        "Hmm, none of those. Ask me something specific.",
    ),
    "terse": (
        "not it. ask me something",
        "nope. what do you need?",
    ),
}

OVERRIDE_TEMPLATES = {
    "paraphrase": (
        "Actually, scrap what I said earlier. What I really need is {constraint}.",
        "Change of plan, forget the earlier preference. It has to be {constraint}.",
        "Let me correct myself: {constraint} is what matters, not what I said before.",
    ),
    "terse": (
        "actually no. {constraint}",
        "scrap that. {constraint}",
    ),
}

BUDGET_TEMPLATES = (
    "my budget is around ${amount}",
    "I'd rather not go far past ${amount}",
    "somewhere around ${amount}",
)

MATERIAL_TEMPLATES = (
    "made from {value}",
    "{value} fabric",
    "it should be {value}",
)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_marketing(text: str) -> str:
    return _squash(PARENTHETICAL_RE.sub("", text)).strip(" -–—:;,.").strip()


def _clauses(text: str) -> list[str]:
    return [part for part in (piece.strip() for piece in CLAUSE_SPLIT_RE.split(text)) if part]


def _apply_synonyms(text: str) -> str:
    return SYNONYM_RE.sub(lambda match: SYNONYMS[match.group(1).lower()], text)


def _budget_phrase(value: str, rng: random.Random) -> str | None:
    match = PRICE_RE.search(value)
    if not match or "budget" not in value.lower():
        return None
    return rng.choice(BUDGET_TEMPLATES).format(amount=round(float(match.group(1))))


def paraphrase_constraint(value: str, rng: random.Random) -> str:
    """Reword one constraint, preserving its content words."""
    budget = _budget_phrase(value, rng)
    if budget:
        return budget
    key_value = KEY_VALUE_RE.match(value)
    if key_value:
        key = key_value.group(1).strip().lower()
        detail = _strip_marketing(key_value.group(2)).lower()
        detail = DEPARTMENTS.get(detail, detail)
        template = DETAIL_PHRASES.get(key, "the {key} should be {value}")
        return _apply_synonyms(template.format(key=key, value=detail)) if detail else value
    stripped = _strip_marketing(value).lower()
    if not stripped:
        return _squash(value).lower()
    if stripped in MATERIALS:
        return rng.choice(MATERIAL_TEMPLATES).format(value=stripped)
    clauses = [clause for clause in (_strip_marketing(item) for item in _clauses(stripped)) if clause]
    if not clauses:
        return _apply_synonyms(stripped)
    kept = clauses[:2]
    if len(kept) > 1 and rng.random() < 0.5:
        kept.reverse()
    return _apply_synonyms(" and ".join(kept))


def terse_constraint(value: str, limit: int = 4) -> str:
    """Reduce one constraint to a short bag of its most concrete tokens."""
    text = PRICE_RE.sub(lambda match: f"${round(float(match.group(1)))}", _strip_marketing(value))
    tokens = [token.strip(".-") for token in NON_WORD_RE.sub(" ", text.lower()).split()]
    tokens = [token for token in tokens if token and token not in STOPWORDS]
    if not tokens:
        return _squash(value).lower()
    salient = {token for token in tokens if token in MATERIALS or COLOR_RE.fullmatch(token)}
    keep = salient | set(tokens[:limit])
    return " ".join(dict.fromkeys(token for token in tokens if token in keep))


def paraphrase_category(category: str, rng: random.Random) -> str:
    lowered = POSSESSIVE_RE.sub(lambda match: match.group(1).lower(), category.lower())
    head = _squash(lowered)
    if rng.random() < 0.5:
        return head
    return f"some kind of {head}" if rng.random() < 0.5 else f"{head} of some sort"


def terse_category(category: str) -> str:
    tokens = [token for token in NON_WORD_RE.sub(" ", category.lower()).split() if token not in STOPWORDS]
    return " ".join(tokens[-2:]) if tokens else category.lower()


@dataclass(frozen=True)
class Persona:
    """Renders simulator decisions as customer-facing text.

    `paraphrase` rewords constraints; `terse` strips them to content tokens.
    Both false reproduces the official simulator wording verbatim.
    """

    name: str
    paraphrase: bool = False
    terse: bool = False

    @property
    def mode(self) -> str:
        return "terse" if self.terse else "paraphrase" if self.paraphrase else "verbatim"

    def _pick(self, bank: dict[str, tuple[str, ...]], rng: random.Random) -> str:
        return rng.choice(bank[self.mode])

    def render_constraint(self, value: str, rng: random.Random) -> str:
        text = paraphrase_constraint(value, rng) if self.paraphrase else str(value)
        return terse_constraint(text) if self.terse else text

    def render_category(self, category: str, rng: random.Random) -> str:
        if self.terse:
            return terse_category(category)
        return paraphrase_category(category, rng) if self.paraphrase else category

    def opening_buying(self, category: str, constraint: str, rng: random.Random) -> str:
        if self.mode == "verbatim":
            return f"I'm looking for {category}. A key requirement is: {constraint}."
        return self._pick(OPENING_TEMPLATES["buying"], rng).format(
            category=self.render_category(category, rng), constraint=self.render_constraint(constraint, rng)
        )

    def opening_browsing(self, category: str, rng: random.Random) -> str:
        if self.mode == "verbatim":
            return f"I'm looking for {category}, but I'm still exploring."
        return self._pick(OPENING_TEMPLATES["browsing"], rng).format(
            category=self.render_category(category, rng)
        )

    def opening_override(self, category: str, old_value: str, rng: random.Random) -> str:
        if self.mode == "verbatim":
            return f"I'm looking for {category}. {old_value}"
        return self._pick(OPENING_TEMPLATES["override"], rng).format(
            category=self.render_category(category, rng), constraint=self.render_constraint(old_value, rng)
        )

    def reveal(self, matches: list[str], rng: random.Random) -> str:
        if self.mode == "verbatim":
            return "For that, what matters is: " + "; ".join(matches) + "."
        rendered = [self.render_constraint(match, rng) for match in matches]
        separator = ", " if self.terse else " and "
        return self._pick(REVEAL_TEMPLATES, rng).format(joined=separator.join(rendered))

    def no_preference(self, attribute: str, rng: random.Random) -> str:
        if self.mode == "verbatim":
            return f"I don't have an additional preference for {attribute}."
        return self._pick(NO_PREFERENCE_TEMPLATES, rng).format(attribute=attribute)

    def boundary(self, attribute: str, rng: random.Random) -> str:
        if self.mode == "verbatim":
            return f"I don't have a preference for {attribute}; please use your judgment."
        return self._pick(BOUNDARY_TEMPLATES, rng).format(attribute=attribute)

    def nudge(self, rng: random.Random) -> str:
        if self.mode == "verbatim":
            return "Those options are not quite right yet. Ask me about one specific attribute."
        return self._pick(NUDGE_TEMPLATES, rng)

    def override(self, official_message: str, new_value: str, rng: random.Random) -> str:
        if self.mode == "verbatim" or not new_value:
            return official_message
        return self._pick(OVERRIDE_TEMPLATES, rng).format(constraint=self.render_constraint(new_value, rng))


PERSONAS: dict[str, Persona] = {
    persona.name: persona
    for persona in (
        Persona("verbatim"),
        Persona("paraphrase", paraphrase=True),
        Persona("terse", terse=True),
        Persona("terse_paraphrase", paraphrase=True, terse=True),
    )
}

CONTROL_PERSONA = "verbatim"
