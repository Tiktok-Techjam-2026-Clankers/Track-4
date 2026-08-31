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
| LLM (gpt-4.1-mini) | Public | 200 | 0.990 | 0.9496 | 3.075 | 0.938393 |
| LLM (gpt-4.1-nano) | Public | 200 | 1.000 | 0.9660 | 3.135 | 0.947104 |

```text
TechnicalScore = 0.50 × Hit@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
```

No sample IDs or target labels are embedded anywhere in `starter/`.

> **Deterministic scores are byte-reproducible; keyed LLM scores are not.** In
> LLM mode the reranker's 3 s no-retry timeout can trip the whole-run fallback
> latch, so a single-process run mixes LLM and deterministic sessions
> non-deterministically. Only the deterministic row reproduces exactly.

### Cost, tokens & latency

Token usage is a feasibility metric, not part of the TechnicalScore.

| Mode | Dataset | Tokens (prompt / completion) | Est. cost¹ | Wall-clock | Per session |
|---|---|---|---:|---:|---:|
| Deterministic | Public (200) | 0 / 0 | **$0.00** | ~152 s total (~39 s one-time index build) | ~0.22 s |
| LLM (gpt-4.1-mini) | Public (200) | 1,249,783 total | ~$0.73 | ~22.4 min | ~6.7 s |
| LLM (gpt-4.1-nano) | Public (200) | 408,879 / 71,595 | ~$0.07² | ~2.8 min² | ~0.83 s² |

¹ Estimated at gpt-4.1-mini list pricing (≈ $0.40 / 1M input, ≈ $1.60 / 1M
output) — **verify current rates**; treat as order-of-magnitude (~$0.002–0.004
per session).
² nano at nano list pricing (≈ $0.10 / 1M input, ≈ $0.40 / 1M output) —
**verify current rates**.

The **deterministic path costs nothing and needs no network** — it is the
default scored path. The LLM layer is an optional enhancement whose feasibility
cost is disclosed above.

> The deterministic pipeline outscores the LLM pipeline — the RRF ranker is
> already strongly tuned and the reranker only sees product titles. The LLM layer
> is retained for its semantic-parsing capability and graceful demonstration of
> conversational intent; deterministic is the default scored path.

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

A single entry point scores the agent: the competition harness in
`evaluator/local_evaluator.py`. It writes `results.json` and prints the metric
summary.

Deterministic — **this is the scored path** (no key, no network):

```bash
DISABLE_LLM=1 python3 -m evaluator.local_evaluator
```

`DISABLE_LLM=1` forces the deterministic pipeline regardless of any key in the
environment. If no `OPENAI_API_KEY` is present, the bare command below is
already deterministic — the env var makes that explicit and reproducible.

LLM mode (requires an OpenAI key in `.env` or `OPENAI_API_KEY`):

```bash
echo "OPENAI_API_KEY=sk-..." > .env             # .env is gitignored
python3 -m evaluator.local_evaluator
```

Both accept `--catalog`, `--dataset`, and `--output` to point at other files.

Run the test suite:

```bash
python3 -m pytest tests/ -q
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
python3 scripts/query.py "waterproof hiking boots under $100"
python3 scripts/query.py --no-llm "a warm wool sweater for winter"   # deterministic
python3 scripts/query.py --top-k 5 --tags casual,outdoors "something for a hike"
python3 scripts/query.py -i                                          # interactive REPL
```

Titles are shown for readability only — the agent never receives `parent_asin`
values in any prompt.

## Enabling / disabling the LLM

Mode is chosen by the **presence of a key**, not a command. To force
deterministic mode use any of:

- `--no-llm` on `scripts/query.py`
- `DISABLE_LLM=1` in the environment
- `Agent(catalog_path, use_llm=False)` in code
- simply leaving `OPENAI_API_KEY` unset

**Whole-run fallback latch:** the first *hard* LLM failure (timeout, network,
HTTP, or empty response) disables the LLM for the rest of the process and drops
to deterministic instantly — a cut network never costs more than one timeout. A
low-confidence answer is not a failure and does not latch.

## Model and fallback rationale

**1. LLM API: `gpt-4.1-mini`.** We use the model for validated intent JSON
and reranking 30 product titles. We chose it because the mini variant reduces
latency relative to full GPT-4.1 while retaining the GPT-4.1 family's
conversational semantic understanding; it is cost-effective for this team due
to lower pricing than the full model and remaining OpenAI credits; and the
team's prior OpenAI API experience reduced integration risk around caching,
schema validation, timeouts, fallback, and token accounting. OpenAI documents
mini as the [smaller, faster GPT-4.1 variant](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Newer models have not been tested under the same harness, so the team decided
that this is a practical selection rationale.

**2. Deterministic path.** Per-session memory and text normalization feed local
BM25, constraint TF-IDF, exact/prefix/fuzzy/override intent-card, category,
phrase, and hashed token/bigram retrieval. Dynamic Reciprocal Rank Fusion and a
turn-aware recommendation ladder balance early rank quality with late coverage;
an optional offline cross-encoder reranks positions 2-30 without moving the
protected top result. Product IDs always come from the catalog, and one hard API
failure latches both LLM calls off while preserving the complete local pipeline.

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

Final results vs baseline (**0.107** TechnicalScore, public 200 sessions):

| Mode | TechnicalScore | Improvement |
|---|---:|---:|
| Deterministic | **0.9707** | +807% |
| LLM (gpt-4.1-mini) | 0.9384 | +777% |
| LLM (gpt-4.1-nano) | 0.9471 | +785% |

## Limitations

- The LLM reranker sees only product titles; richer per-item context (attributes,
  descriptions) could close the gap to the deterministic ranker.
- Clarification is metric-neutral under the current simulator (a generic reply
  reveals a superset of constraints), so candidate-pool-aware questioning is
  deferred — it improves the *human* experience, not the automated score.
- Persona / robustness splits are measured in deterministic mode only.
- Model selection is not exhaustive. Future work will compare newer OpenAI and
  non-OpenAI models under the same prompts, serial runner, timeout/latch policy,
  and datasets, reporting ranking metrics, token use, cost, and p50/p95 latency.

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
scripts/          single-query demo (query.py) + model/catalog fetch helpers
tests/            unit + contract + adversarial tests
docs/             architecture, API contract, implementation notes
evaluator/        local simulator and scorer — the only entry point (do not modify)
data/             catalog + session sets
```

## Team

| Member | Role / contribution |
|---|---|
| Di Heng | Retrieval and ranking pipeline |
| Louis| Conversation memory and intent parsing |
| Xavier | Evaluation and testing |
| Yu Xuan| Devpost writeup |
| Yuhan | Demo video |
