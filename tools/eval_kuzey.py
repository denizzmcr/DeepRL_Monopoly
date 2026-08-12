"""Seat-balanced evaluation of Kuzey's heuristic, parallelised across processes.

WHY THIS FILE EXISTS
`ASU_FROZEN_TEACHER.evaluate` is the seat-balanced evaluator that produced every
number in SUBMISSION.md, but its `parse_agent_spec` only knows scripted/ASU ids
and checkpoint paths -- it cannot seat Kuzey's heuristic. `train.build_opponents`
knows `kuzey`, but its `evaluate` is single-seat and bound to a learning agent.
So this joins the two: the evaluator's *driver* (`_run_game`, unmodified, so the
numbers are comparable to the champion's) with a factory that can also build
Kuzey. Nothing in `monopoly_game_engine/` is touched.

THREE THINGS IT DOES DELIBERATELY

  * PYTHONHASHSEED=0, pinned by re-exec. Kuzey's harness documents that this
    engine is not reproducible without it -- string-hash randomisation flips
    about one game in forty. Our own evaluator never pinned it.

  * THREE WIN-RATE CONVENTIONS, not one. A game that hits the 200-round cap has
    no winner. Our evaluator drops those games from the denominator; Kuzey's
    `play.py` keeps them and scores them as non-wins. On a stalemate-heavy field
    those two conventions are far apart, so both are reported, alongside a third
    that credits the net-worth leader at the cap.

  * COUNTS KUZEY'S INTERNAL FAIL-SAFE. spine.py:347 clamps its own illegal
    actions to END_TURN before returning, so "zero illegal actions" is true by
    construction and says nothing about how often the policy actually went out
    of bounds. The probe below re-runs `_decide` and counts it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Asserted, not self-corrected by re-exec: this project's path contains
# non-ASCII characters that `os.execv` mangles on Windows, which is how Kuzey's
# own `play.py` fails to start here. Set it in the environment before launching.
if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit(
        "PYTHONHASHSEED=0 is required -- this engine is not reproducible without "
        "it. Launch as: PYTHONHASHSEED=0 venv/Scripts/python tools/eval_kuzey.py ..."
    )

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# OUR engine first. Kuzey's `_bind` only claims the `monopoly_game_engine` name
# when nothing else holds it, so importing ours first is what keeps their
# vendored copy out of the process. Asserted in `assert_our_engine` below.
import monopoly_game_engine  # noqa: E402
from monopoly_game_engine.actions import ActionType  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS  # noqa: E402

from ASU_FROZEN_TEACHER.evaluate import (  # noqa: E402
    AgentFactory,
    AgentSpec,
    _run_game,
    checkpoint_sha256,
    parse_agent_spec,
    wilson_interval,
)

END_TURN = int(ActionType.END_TURN)
KUZEY_IDS = ("kuzey", "kuzey-plus")
DEFAULT_MAX_DECISIONS = 20_000


def assert_our_engine() -> str:
    """Fail loudly if Kuzey's vendored engine hijacked the module name."""
    loaded = Path(sys.modules["monopoly_game_engine"].__file__).resolve()
    ours = Path(__file__).resolve().parents[1] / "monopoly_game_engine" / "__init__.py"
    if loaded != ours.resolve():
        raise RuntimeError(
            f"monopoly_game_engine resolves to {loaded}, not our {ours}. "
            "Two engines in one process invalidates every number below."
        )
    return str(loaded)


class KuzeyAdapter:
    """Kuzey's heuristic on a seat, instrumented.

    Calls the same public entry point `train.KuzeyHeuristicOpponent` uses, so
    this measures the policy our league actually plays against. The `_decide`
    probe is observation only -- the public call stays authoritative.
    """

    def __init__(self, player_id: int, variant: str):
        from monopoly_game_engine.train import KuzeyHeuristicOpponent

        self.player_id = player_id
        self.inner = KuzeyHeuristicOpponent(player_id, variant)
        self.decisions = 0
        self.nanoseconds = 0
        self.self_clamps = 0
        self.forced = 0

    def choose_action(self, env) -> int:
        allowed = tuple(int(a) for a in env.get_allowed_actions(self.player_id))
        started = time.perf_counter_ns()
        action = int(self.inner.choose_action(env))
        self.nanoseconds += time.perf_counter_ns() - started
        self.decisions += 1
        if len(allowed) == 1:
            self.forced += 1
        else:
            raw = self.inner.agent._decide(env, self.player_id, allowed, set(allowed))
            if int(raw) not in allowed:
                self.self_clamps += 1
        return action


class KuzeyAwareFactory(AgentFactory):
    """The evaluator's factory, plus the two ids it does not know about."""

    def __init__(self):
        super().__init__()
        self.kuzey_seats: list[KuzeyAdapter] = []

    def build(self, spec: AgentSpec, player_id: int):
        if spec.kind in KUZEY_IDS:
            variant = "plus" if spec.kind == "kuzey-plus" else "champion"
            adapter = KuzeyAdapter(player_id, variant)
            self.kuzey_seats.append(adapter)
            return adapter
        return super().build(spec, player_id)


def parse_spec(value: str) -> AgentSpec:
    normalized = value.strip()
    if normalized in KUZEY_IDS:
        return AgentSpec(normalized, normalized)
    return parse_agent_spec(normalized)


def _job(payload: tuple[tuple[str, ...], int, int, int]) -> dict[str, Any]:
    """One game in a worker process.

    Safe to shard arbitrarily: every game rebuilds its environment from `seed`
    alone, and the heuristic is stateless (EPSILON=0.0, so `decision_seed` is
    dead and `_Context` is rebuilt per decision), so which worker runs which
    game cannot change the result.
    """
    assert_our_engine()
    spec_ids, focus_seat, seed, max_decisions = payload
    specs = tuple(parse_spec(value) for value in spec_ids)
    random.seed(0x5EED ^ (seed * NUM_PLAYERS + focus_seat))
    factory = KuzeyAwareFactory()
    record = _run_game(specs, focus_seat, seed, max_decisions, factory)
    record["kuzey_decisions"] = sum(a.decisions for a in factory.kuzey_seats)
    record["kuzey_nanoseconds"] = sum(a.nanoseconds for a in factory.kuzey_seats)
    record["kuzey_self_clamps"] = sum(a.self_clamps for a in factory.kuzey_seats)
    record["kuzey_forced"] = sum(a.forced for a in factory.kuzey_seats)
    return record


def summarise(identifier: str, games: list[dict[str, Any]]) -> dict[str, Any]:
    """Win rate under all three cap conventions (see module docstring)."""
    seats = [
        (game, seat)
        for game in games
        for seat, policy in enumerate(game["policies"])
        if policy == identifier
    ]
    decisive = [(g, s) for g, s in seats if not g["truncated"]]
    wins = sum(g["winner"] == s for g, s in decisive)
    leads = sum(g["provisional_leader"] == s for g, s in seats)
    total, n_decisive = len(seats), len(decisive)

    def block(w: int, n: int) -> dict[str, Any]:
        low, high = wilson_interval(w, n)
        return {
            "wins": w,
            "games": n,
            "rate_percent": None if n == 0 else 100 * w / n,
            "wilson_95_percent": [100 * low, 100 * high],
        }

    return {
        "decisive_only": block(wins, n_decisive),
        "cap_as_loss": block(wins, total),
        "cap_to_leader": block(leads, total),
        "appearances": total,
        "truncated_appearances": total - n_decisive,
        "mean_final_net_worth": (
            sum(g["final_net_worth"][s] for g, s in seats) / total if total else None
        ),
    }


def load_completed(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    """Games already on disk from an earlier attempt, keyed by (seed, seat)."""
    done: dict[tuple[int, int], dict[str, Any]] = {}
    if path is None or not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done[(record["seed"], record["focus_seat"])] = record
    return done


def run(
    focus: str,
    opponents: tuple[str, str, str],
    seeds: tuple[int, ...],
    workers: int,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    label: str = "",
    jsonl: Path | None = None,
) -> dict[str, Any]:
    specs = (parse_spec(focus), *(parse_spec(o) for o in opponents))

    # An ASU field costs about a minute a game, so a run that loses its work on
    # interruption is an hour thrown away -- which is exactly what happened once.
    # Completed games are appended to `jsonl` as they land and skipped on restart.
    completed = load_completed(jsonl)
    games: list[dict[str, Any]] = list(completed.values())
    jobs = []
    for seed in seeds:
        for focus_seat in range(NUM_PLAYERS):
            if (int(seed), focus_seat) in completed:
                continue
            seats: list[str | None] = [None] * NUM_PLAYERS
            seats[focus_seat] = focus
            others = (s for s in range(NUM_PLAYERS) if s != focus_seat)
            for seat, opponent in zip(others, opponents):
                seats[seat] = opponent
            jobs.append((tuple(s for s in seats if s), focus_seat, int(seed), max_decisions))
    if completed:
        print(f"  [{label}] resuming: {len(completed)} games already on disk, "
              f"{len(jobs)} to go", flush=True)

    started = time.perf_counter()
    total = len(jobs)

    def record_done(record: dict[str, Any], index: int) -> None:
        games.append(record)
        if jsonl is not None:
            with jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if index % 10 == 0 or index == total:
            elapsed = time.perf_counter() - started
            print(
                f"  [{label}] {index:4d}/{total} games  "
                f"{elapsed:6.0f}s  ({elapsed / index:.1f}s/game)",
                flush=True,
            )

    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            record_done(_job(job), index)
    elif jobs:
        # as_completed, not map: map yields in job order, so one slow game holds
        # back every finished record behind it and none of them reach the disk.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                record_done(future.result(), index)
    elapsed = time.perf_counter() - started

    identifiers = sorted({spec.policy_id for spec in specs})
    decisions = sum(g["kuzey_decisions"] for g in games)
    nanoseconds = sum(g["kuzey_nanoseconds"] for g in games)
    return {
        "label": label,
        "focus": specs[0].policy_id,
        "opponents": [spec.policy_id for spec in specs[1:]],
        "seeds": list(seeds),
        "games_played": len(games),
        "truncated_games": sum(g["truncated"] for g in games),
        "round_cap_games": sum(g["rounds"] >= 200 for g in games),
        "win_rates": {name: summarise(name, games) for name in identifiers},
        "kuzey_decisions": decisions,
        "kuzey_ms_per_decision": (nanoseconds / decisions / 1e6) if decisions else None,
        "kuzey_self_clamps": sum(g["kuzey_self_clamps"] for g in games),
        "kuzey_forced_decisions": sum(g["kuzey_forced"] for g in games),
        "scripted_compatibility_fallbacks": sum(
            sum(g["scripted_compatibility_fallbacks"]) for g in games
        ),
        "mean_rounds": sum(g["rounds"] for g in games) / len(games),
        "checkpoint_hashes": {
            spec.policy_id: checkpoint_sha256(spec.checkpoint)
            for spec in specs
            if spec.checkpoint is not None
        },
        "elapsed_seconds": elapsed,
        "engine": assert_our_engine(),
        "games": games,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--focus", default="kuzey")
    parser.add_argument("--opponents", nargs=3, default=("fixed-a", "fixed-b", "fixed-c"))
    parser.add_argument("--seeds", type=int, default=50, help="number of paired seed blocks")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="append each finished game here and resume from it if interrupted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = tuple(range(args.base_seed, args.base_seed + args.seeds))
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        args.focus,
        tuple(args.opponents),
        seeds,
        args.workers,
        args.max_decisions,
        args.label or f"{args.focus} vs {'/'.join(args.opponents)}",
        args.jsonl,
    )
    focus_id = result["focus"]
    summary = result["win_rates"][focus_id]
    print(f"\n{'=' * 72}")
    print(f"  {result['label']}   {result['games_played']} games "
          f"({len(seeds)} seeds x {NUM_PLAYERS} seats)")
    for view in ("decisive_only", "cap_as_loss", "cap_to_leader"):
        block = summary[view]
        low, high = block["wilson_95_percent"]
        rate = block["rate_percent"]
        print(f"  {view:>15}: {block['wins']:4d}/{block['games']:<4d} = "
              f"{'n/a' if rate is None else f'{rate:5.1f}%'}  "
              f"95% CI [{low:.1f}, {high:.1f}]")
    # `truncated` is the 20,000-decision valve, NOT the round cap: env.py:1072
    # sets done=True at max_rounds, so a capped game is a completed game here.
    # Report both, named for what they actually are.
    print(f"  reached the 200-round cap: {result['round_cap_games']}"
          f"/{result['games_played']} (decided on net worth)")
    print(f"  hit the 20k-decision valve: {result['truncated_games']}"
          f"/{result['games_played']}")
    print(f"  kuzey self-clamps: {result['kuzey_self_clamps']} over "
          f"{result['kuzey_decisions']} decisions "
          f"({result['kuzey_ms_per_decision']:.3f} ms each)"
          if result["kuzey_decisions"] else "  kuzey not seated")
    print(f"  elapsed: {result['elapsed_seconds']:.0f}s")
    print("=" * 72)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
