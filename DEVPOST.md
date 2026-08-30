# Shopping Copilot — Devpost Submission

_TechJam 2026 · Track 4 · Conversational E-Commerce Search_

---

## Inspiration

Product search still fails the moment a shopper can't name exactly what they
want. "A comfy waterproof jacket for hiking" returns thousands of results, and
the customer is left refining keywords one at a time. We wanted an agent that
behaves like a good sales associate: it asks the *one* question that narrows the
field fastest, remembers what you've already said, and gets you to the right
product in a handful of turns — not a scroll session.

## What it does

Given an anonymized preference profile and a short message, the agent holds a
multi-turn conversation and, on every turn, returns a ranked Top-10 of catalog
products plus one targeted clarification question. The session succeeds when the
customer's hidden target appears in the Top-10, ideally within the first two or
three turns.

It handles four conversation types the evaluator throws at it — **Buying**,
**Browsing**, **Intent Override** (the shopper changes their mind mid-session),
and **Boundary** behavior — over a frozen 50,000-product Amazon catalog.

**Verified results** (supplied local evaluator, byte-identical across runs):

| Mode | Dataset | Hit@10 | MRR | MTTC | TechnicalScore |
|---|---|---:|---:|---:|---:|
| Deterministic | Public (200) | 1.000 | 0.9933 | 2.580 | **0.966400** |
| Deterministic | Extended (500) | 0.998 | 0.9835 | 2.706 | **0.959927** |

A perfect hit rate on the public set, with the target found on average before
turn 3.

## How we built it

The agent runs a five-stage pipeline every turn:

1. **Intent parsing** — classifies browsing / buying / override and extracts
   structured constraints. Uses `gpt-4.1-mini` when a key is present, with a
   deterministic parser as fallback.
2. **Conversation memory** — tracks active requirements, retired intent after an
   override, declined and already-asked attributes, and a stable recommendation
   ladder, all isolated per session.
3. **Parallel retrieval** — five complementary routes: BM25 (SQLite FTS5),
   local semantic hashed-vectors, constraint TF-IDF, catalog-derived
   *intent-cards* (exact/prefix/fuzzy), and exact-category lookup.
4. **Intent-aware rank fusion** — Reciprocal Rank Fusion with **dynamic weights**
   and an adaptive fusion constant that shift trust toward structured evidence as
   constraints accumulate, and toward semantics right after an override. An
   optional LLM reranker reorders the top-20 by title.
5. **Response policy** — emits the Top-10 and chooses the single most useful next
   question.

**Two modes, one codebase.** Mode is selected purely by the presence of an API
key. The deterministic path needs no network and produces zero tokens; the LLM
path adds semantic understanding on top. A **whole-run fallback latch** means the
first hard LLM failure drops the entire process to deterministic instantly — a
cut network costs at most one timeout, never a per-turn penalty.

**Built with:** Python 3.10, NumPy, SQLite FTS5, OpenAI `gpt-4.1-mini`.
Single runtime dependency (NumPy) for the scored path.

## Challenges we ran into

- **A stale baseline nearly caused a wrong revert.** Old docs quoted a
  `0.9726 / 0.9682` score from a previous (Gemini-era) implementation that was
  never re-measured. We adopted a hard rule — *never trust a score you didn't
  just measure* — and re-verify deterministically after every change. The real
  current numbers are `0.9664 / 0.9599`.
- **The LLM scored *below* the deterministic ranker.** Our RRF pipeline was
  already strongly tuned, and the reranker only sees titles. Rather than force a
  worse-scoring path, we made deterministic the default and kept the LLM as an
  enhancement with clean fallback.
- **Clarification is metric-neutral under the simulator.** We discovered from the
  evaluator that a generic reply reveals a *superset* of constraints, so a clever
  pool-aware question can't beat a generic one on the automated metric. We treat
  proactive clarification as a UX strength to demo, not a score lever to chase.
- **Keeping the pipeline reproducible.** Byte-identical results across runs
  required eliminating every source of nondeterminism in retrieval and fusion.

## Accomplishments we're proud of

- A **1.000 hit rate** on the public set with average time-to-hit under 3 turns.
- A pipeline that is **fully reproducible and offline-capable** — no key, no
  network, no cost — yet upgrades cleanly to an LLM when one is available.
- A genuinely **safe LLM integration**: schema validation, prompt caching,
  per-call timeouts, a whole-run fallback latch, and a strict no-leakage boundary
  (titles and query text only ever reach the model — never product IDs).
- Clean, layered module architecture with a comprehensive test suite.

## What we learned

- Tune the cheap, deterministic path *first* — a well-built classical retriever
  is a very strong baseline that LLMs don't automatically beat.
- Read the evaluator before optimizing: knowing exactly how the simulator reveals
  constraints stopped us from spending effort where the metric couldn't reward it.
- Graceful degradation is a feature. Designing for "the network just died" from
  the start produced a more trustworthy system than bolting on retries later.

## What's next

- Feed the reranker richer per-item context (attributes, descriptions) to close
  the gap with the deterministic ranker.
- Candidate-pool-aware clarification for real users, where a smarter question
  genuinely improves the human experience even if the simulator is indifferent.
- Extend LLM-mode measurement to the persona and robustness splits.

## Try it

```bash
python -m pip install -r requirements.txt
python scripts/evaluate_datasets.py --no-llm     # → 0.9664 / 0.9599, 0 tokens
python -m pytest tests/ -q
```
