# LLM Intent Architecture

## Overview

The shopping copilot uses an LLM-first intent-classification pipeline:

```
User message + compact conversation state
        ↓
OpenAIIntentParser (gpt-4.1-mini, structured JSON)
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

All messages go through the LLM — there is no template detection bypass.
The deterministic fallback only activates when the LLM is unavailable or
returns low confidence.

## Components

### `starter/intent_parser.py`

| Class | Role |
|---|---|
| `IntentResult` | Structured output dataclass — mode, constraints, negations, confidence |
| `IntentParser` | Abstract interface |
| `DeterministicIntentParser` | Minimal fallback — always returns "browsing" |
| `OpenAIIntentParser` | Calls OpenAI API, validates response, caches results |
| `HybridIntentParser` | LLM-first → deterministic fallback |

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
- Counts accumulate per session across all turns.
- The evaluator sums these across sessions for the final report.

## Privacy

- The API key is read from `os.environ` or `.env` — never logged, printed,
  or included in prompts.
- The OpenAI prompt receives only the current message plus a compact
  summary of the conversation state (active query, turn number, session
  mode, last question). It never receives the full transcript, product
  catalogue, or product IDs.
- The LLM cannot inject product IDs — all IDs come from the deterministic
  retrieval pipeline.

## Deterministic Scores (no LLM)

Verified on `di-heng-3` with `OPENAI_API_KEY` unset:

| Dataset | Score |
|---|---|
| Official public | 0.972550 |
| Official extended | 0.966830 |
| Custom verbatim | 0.972550 |
| Custom paraphrase | 0.962214 |
| Custom terse | 0.961100 |
| Custom terse_paraphrase | 0.960750 |

Two consecutive runs produce byte-identical results.
