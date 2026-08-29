# LLM Intent Architecture

## Overview

The shopping copilot uses a hybrid intent-classification pipeline:

```
User message + compact conversation state
        ↓
Template detection (is it a verbatim simulator message?)
        ↓  yes → DeterministicIntentParser (regex)
        ↓  no  ↓
GeminiIntentParser (Gemini 3.5 Flash-Lite, structured JSON)
        ↓  fail → DeterministicIntentParser (fallback)
        ↓
Schema validation
        ↓
Deterministic state reconciliation
        ↓
Existing BM25 + semantic retrieval + structured filters
        ↓
Existing deterministic ranking (with negative constraint penalty)
        ↓
Top-10 response
```

## Components

### `starter/intent_parser.py`

| Class | Role |
|---|---|
| `IntentResult` | Structured output dataclass — mode, constraints, negations, confidence |
| `IntentParser` | Abstract interface |
| `DeterministicIntentParser` | Wraps existing `IntentClassifier` regex logic |
| `GeminiIntentParser` | Calls Gemini API, validates response, caches results |
| `HybridIntentParser` | Template detection → Gemini → deterministic fallback |

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
  "confidence": 0.0-1.0
}

```

## Environment Variable

Set `GEMINI_API_KEY` in your shell environment:

```bash
export GEMINI_API_KEY="your-key-here"
python scripts/score_datasets.py
```

Or load from `.env`:

```bash
export $(grep GEMINI_API_KEY .env | xargs)
python scripts/score_datasets.py
```

Or call `load_dotenv_api_key()` once in your entry-point script before
creating the Agent.

When the variable is unset or empty, the agent falls back to pure
deterministic parsing with zero network calls and zero token usage.

## Failure Behaviour

| Failure | Result |
|---|---|
| `GEMINI_API_KEY` not set | DeterministicIntentParser only — identical to pre-LLM behaviour |
| API call times out (5 s default) | Falls back to deterministic for that turn |
| Gemini returns malformed JSON | Falls back to deterministic for that turn |
| Gemini returns invalid schema | Falls back to deterministic for that turn |
| Network error / HTTP error | Falls back to deterministic for that turn |
| LLM confidence below 0.5 | Ignored; deterministic result used |
| Verbatim simulator template | LLM skipped entirely; deterministic fast path |

Every failure path preserves the deterministic scores exactly.

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
- Token counts come from Gemini's `usageMetadata` response field.
- Counts accumulate per session across all turns.
- The evaluator sums these across sessions for the final report.

## Privacy

- The API key is read from `os.environ` only — never logged, printed, or
  included in prompts.
- The Gemini prompt receives only the current message plus a compact
  summary of the conversation state (active query, turn number, session
  mode, last question). It never receives the full transcript, product
  catalogue, or product IDs.
- The LLM cannot inject product IDs — all IDs come from the deterministic
  retrieval pipeline.

## Deterministic Scores (no LLM)

Verified on `di-heng-2-llm-intent` with `GEMINI_API_KEY` unset:

| Dataset | Score |
|---|---|
| Official public | 0.972550 |
| Official extended | 0.966830 |
| Custom verbatim | 0.972550 |
| Custom paraphrase | 0.962214 |
| Custom terse | 0.961100 |
| Custom terse_paraphrase | 0.960750 |

Two consecutive runs produce byte-identical results.
