# Custom Evaluator 1

A held-out robustness harness that measures how much of the agent's score depends
on the official simulator's exact wording.

## Why it exists

The official simulator builds customer messages out of the target product's own
metadata. `intent_card` slices up to 180 characters of `features` and `details`
straight into the opening line:

```text
I'm looking for Leggings. A key requirement is: 92% Polyester 8% Spandex, high waisted tummy control...
```

That is near-verbatim catalogue text, so lexical overlap between the customer's
words and the target document is unusually high. The competition specification
warns that the hidden set may not read that way:

> If natural-language paraphrasing is added by the organizer, it cannot decide correctness.

This harness estimates the exposure. It keeps the hidden intent card, the ground
truth, and the simulator policy exactly as the organizer defines them, and changes
only how the customer *words* things.

## What is held constant

Personas control phrasing and nothing else. Everything that determines the
information available to the agent is delegated to `evaluator.local_evaluator`:

| Held constant | Source |
| --- | --- |
| Hidden intent card | `materialize_hidden_fields` / `intent_card` |
| Which constraints are revealed, and when | `classify_constraint`, the `disclosed` set |
| Override turn and payload | `behavior_for` |
| Turn budget, hit rule, override gating | mirrored from `evaluate` |
| Recommendation normalisation and metrics | `normalize_recommendations`, `metric_summary` |

Disclosure is tracked on the **raw** constraint string, never the rendered one, so
every persona reveals the same constraints on the same turns. A metric gap between
two personas therefore isolates phrasing sensitivity rather than task difficulty.

## Personas

| Persona | Transformation |
| --- | --- |
| `verbatim` | Reproduces official wording exactly. Calibration control. |
| `paraphrase` | Rewrites `Key: value` details into natural clauses, rounds budgets, compresses multi-clause bullets, reorders, and substitutes meaning-preserving synonyms (`breathable` → `airy`). |
| `terse` | Strips to content tokens — stopwords dropped, marketing tail removed — while always retaining material and colour words. |
| `terse_paraphrase` | Paraphrase, then terse. The hardest setting. |

Paraphrasing is rule-based, offline, and seeded. No model is called, so the suite
adds no dependency, no cost, and no network requirement.

## Determinism

Every utterance is seeded from `f"{persona}\0{sample_id}\0{turn}"`, the same
string-seeded `random.Random` idiom the official evaluator uses. Session ids are
derived (`custom_evaluator_1_{persona}_{sample_id}`) rather than random. Two runs of the same
command produce byte-identical reports.

Latency is the only non-deterministic output and is opt-in behind `--timings`.

## Calibration invariant

The `verbatim` persona must reproduce the official evaluator exactly. This is
asserted three ways in `tests/test_custom_evaluator_1.py`:

1. Opening and reply strings are compared against `initial_message` and
   `customer_reply` directly.
2. A full `run_persona` is compared against `evaluate` on a synthetic catalogue.
3. Published numbers are reproduced on the real agent — `1.0 / 0.990833 /
   2.235 / 0.8765 / 0.972550`, matching `docs/independent_agent_results.json`.

If the control drifts, the harness is wrong and the deltas mean nothing.

## Contract checking

Every response is validated against `docs/agent_api_contract.json` — the enum is
read from the contract file rather than hardcoded, so the check cannot drift from
the spec. Findings are recorded for reporting only; responses still flow through
the official scoring path unchanged, so contract checks never move the metrics.

## Usage

```bash
python -m evaluator.custom_evaluator_1 --shared-agent
```

```bash
python -m evaluator.custom_evaluator_1 --dataset data/private_set.jsonl --limit 100 --transcripts
```

| Flag | Effect |
| --- | --- |
| `--personas` | Subset of `verbatim paraphrase terse terse_paraphrase` |
| `--limit` | Stratified subsample that preserves the scenario mix |
| `--transcripts` | Store per-turn dialogue in the report for failure inspection |
| `--timings` | Record per-turn latency (non-deterministic) |
| `--shared-agent` | Reuse one `Agent` across personas instead of rebuilding per persona |

The report is written to `custom_evaluator_1_results.json` (gitignored) and a comparison table
is printed to stdout.

## Current robustness results

Two byte-identical runs on the 200 public sessions produced:

| Persona | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| `verbatim` | 1.000 | 0.990833 | 2.235 | 0.8765 | **0.972550** |
| `paraphrase` | 1.000 | 0.992381 | 2.775 | 0.8225 | **0.962214** |
| `terse` | 0.995 | 0.991667 | 2.695 | 0.8305 | **0.961100** |
| `terse_paraphrase` | 1.000 | 0.994167 | 2.875 | 0.8125 | **0.960750** |

## Scope

Report-only. It adds no pass/fail gate, modifies no evaluator or agent file, and
stays strictly inside the official session protocol — same turn budget, same
contract, same metrics.
