"""Per-session conversation state and the deterministic fallback classifier.

Holds ``IntentClassifier`` (minimal no-LLM classifier) and
``ConversationMemory`` (short-term state isolated to one evaluator session).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from starter.text_utils import *  # noqa: F401,F403 — shared helpers/constants


class IntentClassifier:
    """Minimal fallback classifier used only when the LLM is unavailable."""

    @staticmethod
    def classify(message: str, active_query: str = "") -> str:
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
    negative_constraints: set[str] = field(default_factory=set)
    suggested_question: str | None = None

    def observe(
        self,
        message: str,
        classifier: IntentClassifier,
        mode_override: str | None = None,
        no_pref_override: bool = False,
    ) -> str:
        self.history.append(message)
        query_before = self.query()
        intent = mode_override if mode_override else classifier.classify(message, query_before)
        if len(self.history) == 1:
            self.session_mode = "buying" if intent == "buying" else "browsing"

        is_no_pref = no_pref_override
        if is_no_pref and self.last_question:
            self.declined_attributes.add(self.last_question)
            if len(self.history) == 2:
                self.boundary_signal = True

        if intent == "override":
            self.previous_intents.append(query_before)
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
        elif not is_no_pref:
            self.active_messages.append(message)

        self.intent = intent
        return intent

    def query(self) -> str:
        return " ".join(self.active_messages)

    def choose_question(self) -> str:
        if (
            self.suggested_question is not None
            and self.suggested_question not in self.declined_attributes
            and self.asked_counts[self.suggested_question] == 0
        ):
            attribute = self.suggested_question
        elif (
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
