# Codebase rules — TechJam Track 4 Shopping Copilot

These rules are load-bearing. They encode competition constraints and traps we
have already hit. Read before changing `starter/` or the docs.

## 1. Preserve the verified scores

The deterministic pipeline is the scored path when the organizer disables the
network. Its verified TechnicalScores on the **current** codebase (HEAD, after
the override-detection + fusion-seeded-coverage fixes) are:

- **Default public (200):** 0.970700  (Hit 1.000, MRR 0.986667, MTTC 2.265)

These **supersede** the earlier 0.966400, which was the score of
the override-*blind* code state (see `docs/agent_implementation.md` §12.5). The
current objective is to **improve** the robust generalization scores *without
dropping* the official score above — so any change to `starter/` must keep the
official score **at or above** 0.970700 and must not regress any
robust scenario. Re-measure after every non-trivial change:

```bash
DISABLE_LLM=1 python3 -m evaluator.local_evaluator   # must print 0.9707
python -m pytest tests/ -q                           # must stay green
```

The robust/stress and split harnesses were removed along with the other
testing evaluators; `evaluator/local_evaluator.py` is now the only scorer and
must stay byte-identical to `main`.

## 2. Never trust a doc number you did not just measure

Older docs quoted a stale `0.9726 / 0.9682` deterministic baseline from the
pre-OpenAI (Gemini-era) code; it was never re-measured and caused a wrong
"regression" revert. **Always re-measure deterministically** (`--no-llm`, or
`DISABLE_LLM=1`, or no `OPENAI_API_KEY`) rather than comparing against a number
written in a file. If you update a results table, run the eval that produced it.

## 3. No leakage — hard constraint

- No public sample IDs, hidden labels, target ASINs, or evaluator answers may
  appear anywhere in `starter/`.
- No target-label or sample-ID memorisation of any kind.
- LLM prompts receive **titles and query text only** — never `parent_asin` /
  product IDs. Product IDs come only from the deterministic retrieval pipeline.
- Never modify files under `evaluator/` to inflate results.

## 4. API key handling

- Read the key from `OPENAI_API_KEY` (env) or `.env` at the project root only.
- **Never** log, print, commit, or place the key in any prompt.
- `.env` must stay gitignored.

## 5. LLM is the default; deterministic is the fallback

- The mode is chosen by the **presence of a key**, not by a command.
- Disable the LLM explicitly with any of: `--no-llm` flag, `DISABLE_LLM=1`, or
  `Agent(catalog_path, use_llm=False)`. All three set the key to `None`.
- **Whole-run latch:** the first *hard* LLM failure (timeout / network / HTTP /
  empty response) disables the LLM for the rest of the process via
  `Agent._latch_llm_off()`. A low-confidence answer is NOT a failure and must
  not latch. Do not add per-turn LLM retries — a cut network must degrade to
  deterministic instantly, not eat a timeout every turn.

## 6. Module layering (`starter/`)

Import direction is strictly downward — never create an upward or cyclic import:

```
text_utils            (leaf: regexes, constants, pure text helpers)
   ├── memory         (IntentClassifier, ConversationMemory)
   ├── retrieval      (BM25 / constraint / card / category / semantic / phrase)
   └── ranking        (LLMReranker, HybridRanker, ResponseBuilder)
          └── agent    (Agent orchestrator; re-exports the public surface)
```

- `agent.py` re-exports the moved classes so `from starter.agent import X`
  keeps working. Keep that surface intact.
- Tests patch where the *callee's* global lives: `starter.agent.load_api_key`
  (Agent calls it) and `starter.ranking.call_openai` (LLMReranker calls it). If
  you move a function, update its patch target.

## 7. Verification checklist before declaring done

1. `python -m pytest tests/ -q` — all green.
2. `DISABLE_LLM=1 python3 -m evaluator.local_evaluator` — 0.970700.
3. If you touched LLM code, note that the LLM path is measured separately and
   currently scores *below* deterministic (≈0.9384 default) — that is expected.
