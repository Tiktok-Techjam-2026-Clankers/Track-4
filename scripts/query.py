"""Run the agent on a single query (or an interactive multi-turn session).

A lightweight manual harness for eyeballing what the agent returns for one
message — the batch evaluators (evaluate_datasets.py etc.) never let you poke a
single query by hand. Mode follows the usual rule: LLM if a key is present,
deterministic otherwise; force deterministic with --no-llm.

    python scripts/query.py "waterproof hiking boots under $100"
    python scripts/query.py --no-llm "a warm wool sweater for winter"
    python scripts/query.py --top-k 5 "running shoes"
    python scripts/query.py --tags casual,outdoors "something for a hike"
    python scripts/query.py -i          # interactive multi-turn REPL

Titles are looked up from the catalog for readability only — the agent itself
never receives parent_asin values in any prompt (leakage rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import Agent  # noqa: E402


CATALOG_PATH = REPO_ROOT / "data" / "catalog.jsonl"
SESSION_ID = "manual-query"


def load_titles(catalog_path: Path) -> dict[str, str]:
    """Map parent_asin -> title for display only (never fed to the agent)."""
    titles: dict[str, str] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            product = json.loads(line)
            titles[str(product.get("parent_asin"))] = str(product.get("title", ""))
    return titles


def print_response(response: dict, titles: dict[str, str]) -> None:
    question = response.get("message", "")
    attribute = response.get("ask_attribute")
    recs = response.get("recommendations", [])
    usage = response.get("usage", {})

    print(f"\n  agent: {question}")
    if attribute and attribute != "null":
        print(f"  (asking about: {attribute})")

    if not recs:
        print("  recommendations: (none this turn)")
    else:
        print(f"  recommendations ({len(recs)}):")
        for i, rec in enumerate(recs, 1):
            asin = rec.get("parent_asin", "")
            score = rec.get("score", 0.0)
            title = titles.get(str(asin), "<title not in catalog>")
            clipped = title if len(title) <= 72 else title[:69] + "..."
            print(f"   {i:>2}. {asin}  {score:.4f}  {clipped}")

    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    print(f"  cumulative tokens: {tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", help="the shopper message to send")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="multi-turn REPL; type 'quit' to exit")
    parser.add_argument("--no-llm", action="store_true",
                        help="force deterministic mode even if a key is present")
    parser.add_argument("--top-k", type=int, default=10, help="recommendations per turn")
    parser.add_argument("--model", default=None,
                        help="override the LLM model (e.g. gpt-4.1-nano)")
    parser.add_argument("--tags", default="",
                        help="comma-separated preference tags for the user profile")
    parser.add_argument("--catalog", default=str(CATALOG_PATH), help="catalog jsonl path")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("provide a query, or pass -i for interactive mode")

    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        raise SystemExit(f"Catalog not found: {catalog_path}")

    print("loading catalog...", end=" ", flush=True)
    agent = Agent(str(catalog_path), use_llm=not args.no_llm, model=args.model)
    mode = f"LLM ({agent.model})" if agent._llm_active else "deterministic"
    print(f"done.\nmode: {mode}")

    titles = load_titles(catalog_path)
    profile: dict = {}
    if args.tags:
        profile["preference_tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    agent.reset(SESSION_ID, profile)

    if not args.interactive:
        print(f"\n  you: {args.query}")
        response = agent.respond(SESSION_ID, args.query, turn=1, top_k=args.top_k)
        print_response(response, titles)
        return

    max_turns = 10
    print(f"interactive mode (max {max_turns} turns) — type 'done' or 'quit' to exit.\n")
    turn = 1
    if args.query:  # allow seeding the REPL with a first query
        print(f"  you: {args.query}")
        print_response(agent.respond(SESSION_ID, args.query, turn, args.top_k), titles)
        turn += 1
    while turn <= max_turns:
        try:
            prompt = "  what are you looking for? " if turn == 1 else "\n  you: "
            message = input(prompt).strip()
        except EOFError:
            print()
            break
        if message.lower() in {"quit", "exit", "done"}:
            print("\nsession ended by user.")
            break
        if not message:
            continue
        print_response(agent.respond(SESSION_ID, message, turn, args.top_k), titles)
        turn += 1
    else:
        print(f"\nmax turns ({max_turns}) reached — session complete.")


if __name__ == "__main__":
    main()
