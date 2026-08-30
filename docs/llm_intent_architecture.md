# LLM Intent Architecture

## Overview

The shopping copilot uses a two-stage LLM pipeline — intent parsing and
semantic reranking — with full deterministic fallback:

```
User message + compact conversation state + user profile
        ↓
OpenAIIntentParser (gpt-4.1-mini, structured JSON)
  → extracts: mode, constraints, confidence, suggested_question
        ↓  fail → DeterministicIntentParser (fallback)
        ↓
Schema validation
        ↓
Deterministic state reconciliation
        ↓
Dynamic RRF weight computation (turn, constraints, override, tags)
        ↓
BM25 + semantic + structured + intent-card retrieval
  (turn-1: preference tags appended as soft retrieval signals)
        ↓
Intent-aware RRF fusion (dynamic weights + adaptive fusion_k)
        ↓
LLMReranker (gpt-4.1-mini, top-20 by title)
  → reorders candidates by semantic relevance
        ↓  fail → keep original RRF ordering
        ↓
Post-fusion overrides, ladder policy
        ↓
Top-10 response
```

All messages go through the LLM — there is no template detection bypass.
The deterministic fallback only activates when the LLM is unavailable or
returns low confidence. Both LLM calls share a common `call_openai()` helper
with SHA-256 prompt caching.

## Components

### `starter/intent_parser.py`

| Class | Role |
|---|---|
| `IntentResult` | Structured output dataclass — mode, constraints, negations, confidence |
| `IntentParser` | Abstract interface |
| `DeterministicIntentParser` | Minimal fallback — always returns "browsing" |
| `OpenAIIntentParser` | Calls OpenAI API, validates response, caches results |
| `HybridIntentParser` | LLM-first → deterministic fallback |

### `starter/agent.py` (LLM components)

| Class / Function | Role |
|---|---|
| `call_openai()` | Shared OpenAI chat completions helper (used by both parser and reranker) |
| `LLMReranker` | Reranks top-20 fused candidates via LLM, falls back to original order |
| `HybridRanker.compute_weights()` | Dynamic RRF weight computation from session state |

### Schema

```json
{
  "mode": "browsing | buying | override",
  "operation": "add | replace | remove | none",
  "category": "string or null",
  "add_constraints": {"attribute": "value"},
  "remove_constraints": ["removed preferences"],
  "negative_constraints": ["excluded items"],
  "no_preference": ["attributes without preference"],
  "referenced_previous_item": true/false,
  "reference_description": "string or null",
  "confidence": 0.0-1.0,
  "suggested_question": "material | color | style | ... | other | null"
}

```

## Environment Variable

Place your key in `.env` at the project root:

```
OPENAI_API_KEY=your-key-here
```

The agent reads `.env` automatically — no manual `export` needed.
You can also set the variable in your shell if you prefer:

```bash
export OPENAI_API_KEY="your-key-here"
python scripts/evaluate_datasets.py
```

When the variable is unset or empty, the agent falls back to pure
deterministic parsing with zero network calls and zero token usage.

## Failure Behaviour

| Failure | Result |
|---|---|
| `OPENAI_API_KEY` not set | DeterministicIntentParser only |
| API call times out (5 s default) | Falls back to deterministic for that turn |
| OpenAI returns malformed JSON | Falls back to deterministic for that turn |
| OpenAI returns invalid schema | Falls back to deterministic for that turn |
| Network error / HTTP error | Falls back to deterministic for that turn |
| LLM confidence below 0.5 | Ignored; deterministic result used |

Every failure path preserves the deterministic fallback.

## Token Accounting

Each response includes cumulative session token usage:

```json
{
  "usage": {
    "prompt_tokens": 340,
    "completion_tokens": 110
  }
}
```

- When the LLM is disabled (no key), both fields are 0.
- Token counts come from OpenAI's `usage` response field.
- Counts accumulate per session across all turns from both LLM calls (intent + reranker).
- The evaluator sums these across sessions for the final report.
- Typical per-turn cost: ~200 prompt + ~50 completion (intent) + ~200 prompt + ~50 completion (rerank).

## Failure Behaviour — LLM Reranker

| Failure | Result |
|---|---|
| `OPENAI_API_KEY` not set | Reranker not created; RRF ordering used as-is |
| API call times out (3 s default) | Original RRF ranking preserved |
| LLM returns invalid JSON or missing `order` key | Original RRF ranking preserved |
| LLM returns out-of-range indices | Missing items appended in original order |
| Single candidate or empty list | Returned unchanged (no API call made) |

## Privacy

- The API key is read from `os.environ` or `.env` — never logged, printed,
  or included in prompts.
- The **intent parser** prompt receives the current message plus a compact
  summary of the conversation state (active query, turn number, session
  mode, last question, user profile summary, preference tags). It never
  receives the full transcript, product catalogue, or product IDs.
- The **reranker** prompt receives numbered product titles (truncated to
  120 chars) plus the query and user profile. It never receives ASINs or
  internal identifiers.
- The LLM cannot inject product IDs — all IDs come from the deterministic
  retrieval pipeline.

## Deterministic Scores (no LLM)

Re-measured on the current OpenAI codebase with `OPENAI_API_KEY` unset. (Earlier
revisions quoted 0.9726/0.9668 and custom-persona rows; those predate the
Gemini→OpenAI switch and were never re-measured — superseded here.)

| Dataset | Hit@10 | MRR | MTTC | Score |
|---|---|---|---|---|
| Official public (200) | 1.000 | 0.993333 | 2.580 | 0.966400 |
| Official extended (500) | 0.998 | 0.983490 | 2.706 | 0.959927 |

Two consecutive runs produce byte-identical results. The custom persona splits
have not yet been re-measured on the OpenAI codebase and are omitted rather than
quoted stale.

## LLM Scores (OpenAI gpt-4.1-mini, intent parsing + semantic reranking)

Measured live on the default public set (200 sessions), two LLM calls per turn,
with the reranker gating + wide-window guards enabled:

| Dataset | Hit@10 | MRR | MTTC | Score | vs Deterministic |
|---|---|---|---|---|---|
| Default public (200) | 0.990 | 0.9496 | 3.075 | 0.938393 | -0.028 |

Run cost: ~22.4 min wall-clock, 1,249,783 tokens (1,066,281 prompt +
183,502 completion). The two guards cut ~170k tokens and lifted the score from
an ungated 0.934214.

The LLM-enabled score sits below the deterministic 0.966400 baseline: the
deterministic RRF ranker is already strongly tuned, and the reranker sees only
product titles. The full LLM pipeline is retained for its semantic-ranking
capability; deterministic mode is the higher-scoring, zero-cost fallback.
Extended (500) and persona splits have only been measured in deterministic
mode (see above) — the LLM pipeline was not re-run on them due to cost.
