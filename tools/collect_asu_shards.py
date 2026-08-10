"""
collect_asu_shards.py
---------------------
Collects ASU-labelled decisions in parallel shards for policy distillation.

`monopoly_bench.training.collect_asu_examples` is single-process and takes about
62 seconds per game on an M4 Pro, so a useful dataset takes hours serially. Games
are independent, so this fans them out across processes and writes one `.npz`
shard per worker.

Shards are named after their opponent mix and seed range, and an existing shard
file is skipped, so an interrupted run resumes by re-running the same command.

Seed ranges are allocated disjointly across every mix in a run, and each row
records its game's seed, so shards from different runs and different machines can
be merged without colliding — provided each machine uses its own --seed-base.

Usage:
    # default 50/30/20 mix over 600 games on 10 workers
    python tools/collect_asu_shards.py --games 600 --seed-base 100000

    # explicit mixes: "a,b,c:count"
    python tools/collect_asu_shards.py --seed-base 500000 \
        --mix fixed-a,fixed-b,fixed-c:200 \
        --mix asu-value-v1,asu-value-v1,asu-value-v1:100

    python tools/collect_asu_shards.py --games 20 --seed-base 1 --dry-run
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_bench.engine import NUM_PLAYERS  # noqa: E402
from monopoly_bench.training import (  # noqa: E402
    OPPONENT_IDS,
    collect_asu_examples,
    save_asu_examples,
)

# Fraction of games per opponent mix when --mix is not given. Fixed-A/B/C is the
# matchup the student is evaluated on; D/E/F and ASU add states those three never
# produce, which is the opponent-diversity guardrail in CLAUDE.md section 10.
DEFAULT_MIXES = (
    (("fixed-a", "fixed-b", "fixed-c"), 0.5),
    (("fixed-d", "fixed-e", "fixed-f"), 0.3),
    (("asu-value-v1", "asu-value-v1", "asu-value-v1"), 0.2),
)


def parse_mix(text: str) -> tuple[tuple[str, ...], int]:
    """Parse ``fixed-a,fixed-b,fixed-c:200`` into the mix and its game count."""
    specification, separator, count = text.rpartition(":")
    if not separator:
        raise argparse.ArgumentTypeError(
            f"--mix needs a game count, e.g. {text},...:200"
        )
    opponents = tuple(name.strip() for name in specification.split(","))
    if len(opponents) != 3:
        raise argparse.ArgumentTypeError(
            f"--mix needs exactly 3 comma-separated opponents, got {len(opponents)}"
        )
    unknown = [name for name in opponents if name not in OPPONENT_IDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown opponent(s) {unknown}; expected from {OPPONENT_IDS}"
        )
    try:
        games = int(count)
    except ValueError:
        raise argparse.ArgumentTypeError(f"game count must be an integer: {count!r}")
    if games < 1:
        raise argparse.ArgumentTypeError("game count must be positive")
    return opponents, games


def mix_label(opponents: tuple[str, ...]) -> str:
    """Short filesystem-safe name for a mix, collapsing three identical seats."""
    if len(set(opponents)) == 1:
        return f"3x{opponents[0]}"
    return "-".join(name.replace("fixed-", "") for name in opponents)


def build_plan(mixes, *, seed_base, games_per_shard, output_dir):
    """Allocate disjoint seed ranges and one shard task per chunk of games.

    Seat balance comes from covering whole blocks of NUM_PLAYERS consecutive
    seeds, so every mix's game count must be a multiple of NUM_PLAYERS. A mix of
    10 games would leave two seats over-represented in the merged dataset.
    """
    for opponents, games in mixes:
        if games % NUM_PLAYERS:
            raise ValueError(
                f"mix {','.join(opponents)} has {games} games, which is not a "
                f"multiple of {NUM_PLAYERS}; seat balance would be skewed"
            )
    if seed_base % NUM_PLAYERS:
        raise ValueError(
            f"--seed-base must be a multiple of {NUM_PLAYERS} so each mix starts "
            "on a seat boundary"
        )
    plan = []
    cursor = seed_base
    for opponents, games in mixes:
        label = mix_label(opponents)
        remaining = games
        while remaining > 0:
            size = min(games_per_shard, remaining)
            name = f"asu_{label}_{cursor:09d}_{size:04d}.npz"
            plan.append(
                {
                    "output": str(Path(output_dir) / name),
                    "games": size,
                    "seed_base": cursor,
                    "opponents": list(opponents),
                }
            )
            cursor += size
            remaining -= size
    return plan


def collect_shard(task: dict) -> dict:
    """Collect one shard. Returns a result record; never raises past the pool."""
    output = Path(task["output"])
    started = time.perf_counter()
    if output.exists():
        return {**task, "status": "skipped", "rows": 0, "seconds": 0.0}
    try:
        examples = collect_asu_examples(
            games=task["games"],
            seed_base=task["seed_base"],
            max_rounds=task["max_rounds"],
            opponents=task["opponents"],
        )
        save_asu_examples(output, examples)
    except Exception as exc:  # one bad shard must not lose the rest of the run
        return {
            **task,
            "status": "failed",
            "rows": 0,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **task,
        "status": "collected",
        "rows": int(len(examples["states"])),
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect ASU-labelled decisions in parallel shards"
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "artifacts" / "asu_shards")
    )
    parser.add_argument(
        "--games",
        type=int,
        default=600,
        help="total games, split across the default mixes (ignored if --mix given)",
    )
    parser.add_argument(
        "--mix",
        action="append",
        type=parse_mix,
        default=None,
        help="explicit mix as 'opp1,opp2,opp3:games'; repeatable",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        required=True,
        help="first game seed; use a disjoint range per machine",
    )
    parser.add_argument(
        "--games-per-shard",
        type=int,
        default=12,
        help="games per shard; a multiple of 4 keeps every shard seat-balanced",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mix:
        mixes = args.mix
    else:
        if args.games < 1:
            parser.error("--games must be positive")
        if args.games % NUM_PLAYERS:
            parser.error(
                f"--games must be a multiple of {NUM_PLAYERS} for seat balance"
            )
        mixes = []
        assigned = 0
        for index, (opponents, share) in enumerate(DEFAULT_MIXES):
            # Every share is rounded down to a whole seat block; the last mix
            # absorbs the remainder so the total is exact and still balanced.
            count = (
                args.games - assigned
                if index == len(DEFAULT_MIXES) - 1
                else int(args.games * share) // NUM_PLAYERS * NUM_PLAYERS
            )
            if count > 0:
                mixes.append((opponents, count))
                assigned += count

    plan = build_plan(
        mixes,
        seed_base=args.seed_base,
        games_per_shard=args.games_per_shard,
        output_dir=args.output_dir,
    )
    for task in plan:
        task["max_rounds"] = args.max_rounds

    total_games = sum(task["games"] for task in plan)
    print(f"{len(plan)} shards, {total_games} games, {args.workers} workers")
    for opponents, games in mixes:
        print(f"  {games:5d} games vs {', '.join(opponents)}")
    print(f"  seeds {args.seed_base}..{args.seed_base + total_games - 1}")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    # spawn is the macOS default and keeps workers free of inherited torch state
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        for done, result in enumerate(
            pool.imap_unordered(collect_shard, plan), start=1
        ):
            results.append(result)
            elapsed = time.perf_counter() - started
            # Shards finish in waves of --workers, so dividing completions by
            # elapsed time badly overestimates the remaining time during the
            # first wave. Project from mean shard duration and how many can run
            # concurrently instead.
            durations = [r["seconds"] for r in results if r["status"] == "collected"]
            mean_shard = sum(durations) / len(durations) if durations else 0.0
            waves_left = (len(plan) - done + args.workers - 1) // args.workers
            remaining = waves_left * mean_shard
            note = result.get("error", f"{result['rows']} rows")
            print(
                f"[{done}/{len(plan)}] {result['status']:9s} "
                f"{Path(result['output']).name} — {note} "
                f"({result['seconds']:.0f}s, ~{remaining / 60:.0f} min left)",
                flush=True,
            )

    collected = [r for r in results if r["status"] == "collected"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    rows = sum(r["rows"] for r in collected)
    elapsed = time.perf_counter() - started

    print(f"\n{'=' * 60}")
    print(f"  Shards collected : {len(collected)}")
    print(f"  Shards skipped   : {len(skipped)} (already present)")
    print(f"  Shards failed    : {len(failed)}")
    print(f"  Rows this run    : {rows}")
    print(f"  Wall time        : {elapsed:.0f}s")
    if collected:
        games_done = sum(r["games"] for r in collected)
        print(f"  Throughput       : {elapsed / games_done:.1f}s/game wall")
    print(f"{'=' * 60}")
    for failure in failed:
        print(f"FAILED {Path(failure['output']).name}: {failure['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
