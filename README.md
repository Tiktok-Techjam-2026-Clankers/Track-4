# ShopMind Shopping Copilot — TechJam 2026 Track 4

A conversational product-search agent that asks useful follow-up questions and
surfaces the customer's hidden target product within 10 turns. Built on a frozen
50,000-product Amazon catalog (`Clothing_Shoes_and_Jewelry`, Amazon Reviews 2023).

The agent runs in two interchangeable modes selected automatically by the
presence of an API key:

- **Deterministic mode** (no key) — a fully local BM25 + semantic + structured
  retrieval and rank-fusion pipeline. Zero network, zero tokens. This is the
  higher-scoring, always-available path.
- **LLM mode** (key present) — the same pipeline with an OpenAI `gpt-4.1-mini`
  layer for intent parsing and semantic reranking. It degrades to deterministic
  instantly and permanently on the first network failure.

## Verified results

Scored with the supplied local evaluator against the frozen catalog. Two
consecutive deterministic runs are byte-identical.

| Mode | Dataset | Sessions | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---:|---:|---:|---:|---:|
| Deterministic | Public | 200 | 1.000 | 0.9867 | 2.265 | **0.970700** |
| Deterministic | Extended holdout | 500 | 0.998 | 0.9886 | 2.370 | **0.968190** |
| LLM (gpt-4.1-mini) | Public | 200 | 0.990 | 0.9496 | 3.075 | 0.938393 |
| LLM (gpt-4.1-mini) | Extended holdout | 500 | 0.940 | 0.9080 | 3.692 | 0.888563 |
| LLM (gpt-4.1-nano) | Public | 200 | 1.000 | 0.9660 | 3.135 | 0.947104 |
| LLM (gpt-4.1-nano) | Extended holdout | 500 | 0.942 | 0.9083 | 3.656 | 0.890368 |

```text
TechnicalScore = 0.50 × Hit@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
```

The extended holdout is an additional local set, not used by the agent at
runtime. No sample IDs or target labels are embedded anywhere in `starter/`.

> **Deterministic scores are byte-reproducible; keyed LLM scores are not.** In
> LLM mode the reranker's 3 s no-retry timeout can trip the whole-run fallback
> latch, so a single-process run mixes LLM and deterministic sessions
> non-deterministically. The LLM extended-holdout row above is a *forced
> full-coverage* measurement (all 500 sessions LLM-driven); a plain keyed run
> will land somewhere between it and the deterministic score. Only the
> deterministic rows reproduce exactly.

### Cost, tokens & latency

Token usage is a feasibility metric, not part of the TechnicalScore.

| Mode | Dataset | Tokens (prompt / completion) | Est. cost¹ | Wall-clock | Per session |
|---|---|---|---:|---:|---:|
| Deterministic | Public + Extended (700) | 0 / 0 | **$0.00** | ~191 s total (~39 s one-time index build) | ~0.22 s |
| LLM (gpt-4.1-mini) | Public (200) | 1,249,783 total | ~$0.73 | ~22.4 min | ~6.7 s |
| LLM (gpt-4.1-mini) | Extended (500) | 1,101,929 / 201,125 | ~$0.76 | ~73.6 min² | ~8.8 s² |
| LLM (gpt-4.1-nano) | Public (200) | 408,879 / 71,595 | ~$0.07³ | ~2.8 min³ | ~0.83 s³ |
| LLM (gpt-4.1-nano) | Extended (500) | 1,119,223 / 199,075 | ~$0.19³ | ~7.2 min³ | ~0.86 s³ |

¹ Estimated at gpt-4.1-mini list pricing (≈ $0.40 / 1M input, ≈ $1.60 / 1M
output) — **verify current rates**; treat as order-of-magnitude (~$0.002–0.004
per session).
² Extended LLM wall-clock and per-session time are inflated by 7 Agent rebuilds
(each reloads the catalog) used to force full LLM coverage past the fallback
latch; pure eval time is lower.
³ nano measured differently: 8 parallel workers with a 20 s timeout and the
whole-run latch neutralised on the measurement instance (no `starter/` change),
so wall-clock/per-session are far lower than the serially-rebuilt mini rows and
not directly comparable. Cost at nano list pricing (≈ $0.10 / 1M input, ≈ $0.40
/ 1M output) — **verify current rates**.

The **deterministic path costs nothing and needs no network** — it is the
default scored path. The LLM layer is an optional enhancement whose feasibility
cost is disclosed above.

> The deterministic pipeline outscores the LLM pipeline on both sets — the RRF
> ranker is already strongly tuned and the reranker only sees product titles. The
> LLM layer is retained for its semantic-parsing capability and graceful
> demonstration of conversational intent; deterministic is the default scored path.

## Setup

Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt                # numpy (required) + fastembed (optional)
```

The deterministic pipeline runs on **numpy alone**. `fastembed` powers the
optional offline cross-encoder reranker (`starter.ranking.LocalReranker`); if it
or its weights are absent, the reranker degrades to identity and scores are
unchanged.

Download the catalog from the GitHub Release attached to this repo:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl            # verify against SHA256SUMS
```

## Reproduce the scores

Deterministic (no key needed — this is the scored path):

```bash
python scripts/evaluate_datasets.py --no-llm
# → Default 0.9707 · Extended 0.9682 · 0 tokens
```

LLM mode (requires an OpenAI key in `.env` or `OPENAI_API_KEY`):

```bash
echo "OPENAI_API_KEY=sk-..." > .env             # .env is gitignored
python scripts/evaluate_datasets.py
```

Run the test suite:

```bash
python -m pytest tests/ -q
```

## Try a single query

Run the agent on one message, or start an interactive multi-turn session. Mode
follows the same key-presence rule; force deterministic with `--no-llm`.

A single-query run executes one turn — the agent returns its top recommendations
and may ask a clarifying question (e.g. budget, material). Use `-i` for an
interactive session where you can answer follow-ups across up to 10 turns.
Type `done` or `quit` to end the session early if you're satisfied with the
recommendations.

```bash
python scripts/query.py "waterproof hiking boots under $100"
python scripts/query.py --no-llm "a warm wool sweater for winter"   # deterministic
python scripts/query.py --top-k 5 --tags casual,outdoors "something for a hike"
python scripts/query.py -i                                          # interactive REPL
```

Titles are shown for readability only — the agent never receives `parent_asin`
values in any prompt.

## Enabling / disabling the LLM

Mode is chosen by the **presence of a key**, not a command. To force
deterministic mode use any of:

- `--no-llm` on the evaluation scripts
- `DISABLE_LLM=1` in the environment
- `Agent(catalog_path, use_llm=False)` in code
- simply leaving `OPENAI_API_KEY` unset

**Whole-run fallback latch:** the first *hard* LLM failure (timeout, network,
HTTP, or empty response) disables the LLM for the rest of the process and drops
to deterministic instantly — a cut network never costs more than one timeout. A
low-confidence answer is not a failure and does not latch.

## Architecture

Every turn runs five stages. The LLM layer is optional at two points (marked ★).

```
User message + compact conversation state + anonymized profile
        │
        ▼
1. Intent parsing        ★ gpt-4.1-mini (structured JSON) → deterministic fallback
        │                   mode · constraints · confidence · suggested_question
        ▼
2. Conversation memory      active requirements, retired intent, declined
                            attributes, asked questions, recommendation ladder
        │
        ▼
3. Retrieval (parallel)     BM25 (FTS5) · semantic hashed-vectors · constraint
                            TF-IDF · catalog intent-cards · exact category
        │
        ▼
4. Rank fusion              intent-aware RRF with dynamic weights + adaptive
                            fusion-k · popularity tie-break · late-turn exploration
        │
        ▼
   Semantic rerank       ★ gpt-4.1-mini reorders top-30 by title → RRF order on fail
                            (offline: LocalReranker cross-encoder, same seam)
        │
        ▼
5. Response policy          Top-10 + one clarification question, then repeat
```

### Module layout (`starter/`)

Import direction is strictly downward.

| Module | Responsibility |
|---|---|
| [text_utils.py](starter/text_utils.py) | regexes, constants, pure text helpers (leaf) |
| [memory.py](starter/memory.py) | `IntentClassifier`, `ConversationMemory` |
| [retrieval.py](starter/retrieval.py) | BM25 / constraint / intent-card / category / semantic / phrase routes |
| [intent_parser.py](starter/intent_parser.py) | deterministic + OpenAI + hybrid intent parsers |
| [ranking.py](starter/ranking.py) | `LLMReranker`, `LocalReranker`, `HybridRanker`, `ResponseBuilder` |
| [agent.py](starter/agent.py) | orchestrator; re-exports the public surface |

The semantic encoder and every index are built locally from visible frozen
catalog fields. LLM prompts receive **titles and query text only** — never
`parent_asin` or product IDs. All product IDs come from the deterministic
retrieval pipeline.

## Key design decisions

- **Deterministic-first.** The scored path needs no key, no network, and is
  reproducible byte-for-byte. The LLM is an enhancement, never a dependency.
- **Intent-aware RRF with dynamic weights.** Fusion weights and `fusion_k` adapt
  to turn number, constraint count, and intent overrides, so late, constraint-rich
  turns trust structured evidence more and fresh overrides lean on semantics.
- **Catalog intent-cards.** Structured constraint clauses derived from the
  catalog let the agent match disclosed requirements exactly, with fuzzy and
  prefix fallbacks for paraphrases — no evaluator templates or hidden labels.
- **Proactive clarification.** The agent asks one targeted follow-up per turn,
  prioritising an LLM-suggested attribute, then an open question, then a fixed
  attribute sequence — without repeating declined or answered attributes.
- **Safe LLM integration.** SHA-256 prompt caching, strict schema validation,
  per-call timeouts, and the whole-run fallback latch keep the LLM path from ever
  degrading the deterministic guarantee.

## Agent interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...

    def respond(self, session_id, user_message, turn, top_k) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}, ...],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
```

`ask_attribute` ∈ {`category`, `material`, `color`, `size`, `style`, `brand`,
`budget`, `feature`, `use_case`, `other`, `null`}. See
[docs/agent_api_contract.json](docs/agent_api_contract.json).

## Improvements over baseline

The organizer-provided weak BM25 baseline scores **0.107** TechnicalScore on the
200 public sessions (Hit@10 0.125, MRR 0.068, MTTC 9.81). Key improvements:

| Improvement | Effect |
|---|---|
| Multi-route retrieval (BM25 + constraint + intent-card + category + semantic) | Hit@10 from 0.125 to ~0.95; single BM25 misses structured constraints entirely |
| Intent-aware RRF with dynamic weights | Fuses complementary routes; adapts trust per turn and constraint count |
| Conversation memory + intent override detection | Handles mid-session mind-changes; prevents stale constraints from dominating |
| Catalog-derived intent-cards with fuzzy/prefix fallback | Exact + approximate matching of disclosed attributes against catalog fields |
| Fusion-seeded late-turn exploration | Lifts coverage on hard turn-10 boundary sessions |
| Proactive clarification strategy | Asks one targeted follow-up per turn, reducing average MTTC from 9.81 to 2.27 |
| LLM intent parsing (gpt-4.1-mini) | Structured JSON extraction of mode, constraints, and confidence; falls back to deterministic on failure |
| LLM semantic reranking | Reorders top candidates by title relevance; optional local cross-encoder alternative via fastembed |
| Whole-run fallback latch | Instant, permanent degradation to deterministic on first network failure — no per-turn penalty |

Final results vs baseline (**0.107** TechnicalScore):

| Mode | Public (200) | Extended (500) |
|---|---:|---:|
| Deterministic | **0.9707** (+807%) | **0.9682** |
| LLM (gpt-4.1-mini) | 0.9384 (+777%) | 0.8886 |
| LLM (gpt-4.1-nano) | 0.9471 (+785%) | 0.8904 |

## Limitations & future work

- The LLM reranker sees only product titles; richer per-item context (attributes,
  descriptions) could close the gap to the deterministic ranker.
- Clarification is metric-neutral under the current simulator (a generic reply
  reveals a superset of constraints), so candidate-pool-aware questioning is
  deferred — it improves the *human* experience, not the automated score.
- Persona / robustness splits are measured in deterministic mode only.

## Security & data handling

- API keys are read only from `OPENAI_API_KEY` or a gitignored `.env`; never
  logged, printed, committed, or placed in any prompt.
- No public sample IDs, hidden labels, target ASINs, or evaluator answers appear
  in `starter/`. The evaluator is never modified to inflate results.
- Catalog and sessions derive from Amazon Reviews 2023 (McAuley Lab, UCSD). See
  [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) before using or redistributing.

## Repository map

```text
starter/          agent implementation (deterministic + optional LLM)
scripts/          evaluation entry points (evaluate_datasets.py is the main one)
tests/            unit + contract + adversarial tests
docs/             architecture, API contract, implementation notes
evaluator/        local simulator and scorer (do not modify)
data/             catalog + session sets
```

## Team

| Member | Role / contribution |
|---|---|
| _TODO_ | _fill in before submission_ |
