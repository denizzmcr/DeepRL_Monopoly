"""Post-process the JSONs from `eval_kuzey.py` into the table for KUZEY_EVAL.md.

WHY THIS IS SEPARATE FROM THE MEASUREMENT

`_run_game` marks a game `truncated` when `not env.done`, and it is easy to read
that as "hit the 200-round cap". It is not. `env._check_game_over` (env.py:1072)
sets `done = True` when one player remains **or** `round >= max_rounds`, so a
capped game is `done` and never truncated; `truncated` only catches the 20,000-
decision safety valve. Every seat-balanced number this project has quoted
therefore already counts capped games, scored by `env.winner()`, which falls
through to the highest net worth.

That distinction matters more than it sounds. Winning outright means bankrupting
the other three. Winning at the cap means being ahead on an accounting measure
when the clock stopped. Against an unknown rival those are not the same asset,
so this splits them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CAP_ROUNDS = 200


def wilson(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    rate = wins / games
    denom = 1 + z * z / games
    centre = (rate + z * z / (2 * games)) / denom
    radius = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denom
    return (100 * max(0.0, centre - radius), 100 * min(1.0, centre + radius))


def breakdown(games: list[dict[str, Any]], identifier: str) -> dict[str, Any]:
    seats = [
        (game, seat)
        for game in games
        for seat, policy in enumerate(game["policies"])
        if policy == identifier
    ]
    total = len(seats)
    outright = sum(g["winner"] == s and g["rounds"] < CAP_ROUNDS for g, s in seats)
    at_cap = sum(g["winner"] == s and g["rounds"] >= CAP_ROUNDS for g, s in seats)
    wins = outright + at_cap
    low, high = wilson(wins, total)
    per_seat: dict[int, list[int]] = {}
    for game, seat in seats:
        bucket = per_seat.setdefault(seat, [0, 0])
        bucket[1] += 1
        if game["winner"] == seat:
            bucket[0] += 1
    return {
        "appearances": total,
        "wins": wins,
        "win_rate": 100 * wins / total if total else None,
        "wilson_95": [low, high],
        "outright_wins": outright,
        "cap_wins": at_cap,
        "share_of_wins_at_cap": 100 * at_cap / wins if wins else None,
        "mean_net_worth": (
            sum(g["final_net_worth"][s] for g, s in seats) / total if total else None
        ),
        "wins_by_seat": {seat: value for seat, value in sorted(per_seat.items())},
    }


def report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    games = payload["games"]
    capped = sum(g["rounds"] >= CAP_ROUNDS for g in games)
    identifiers = sorted({p for g in games for p in g["policies"]})
    result = {
        "label": payload.get("label") or path.stem,
        "focus": payload["focus"],
        "opponents": payload["opponents"],
        "games": len(games),
        "capped_games": capped,
        "capped_percent": 100 * capped / len(games) if games else 0.0,
        "decision_valve_truncations": sum(g["truncated"] for g in games),
        "mean_rounds": sum(g["rounds"] for g in games) / len(games),
        "kuzey_ms_per_decision": payload.get("kuzey_ms_per_decision"),
        "kuzey_self_clamps": payload.get("kuzey_self_clamps"),
        "kuzey_decisions": payload.get("kuzey_decisions"),
        "scripted_fallbacks": payload.get("scripted_compatibility_fallbacks"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "policies": {name: breakdown(games, name) for name in identifiers},
    }
    return result


def render(result: dict[str, Any]) -> str:
    lines = [
        f"### {result['label']}",
        f"focus={result['focus']}  opponents={', '.join(result['opponents'])}  "
        f"N={result['games']}",
        f"capped at {CAP_ROUNDS} rounds: {result['capped_games']}/{result['games']} "
        f"({result['capped_percent']:.1f}%)   mean rounds {result['mean_rounds']:.1f}",
        "",
        f"{'policy':<34} {'win%':>7} {'95% CI':>16} {'outright':>9} {'at cap':>7}",
    ]
    for name, data in sorted(
        result["policies"].items(), key=lambda kv: -(kv[1]["win_rate"] or 0)
    ):
        low, high = data["wilson_95"]
        lines.append(
            f"{name[:34]:<34} {data['win_rate']:>6.1f}% "
            f"{f'[{low:.1f}, {high:.1f}]':>16} "
            f"{data['outright_wins']:>9} {data['cap_wins']:>7}"
        )
    if result["kuzey_decisions"]:
        lines += [
            "",
            f"kuzey: {result['kuzey_decisions']} decisions, "
            f"{result['kuzey_ms_per_decision']:.3f} ms each, "
            f"{result['kuzey_self_clamps']} internal fail-safe clamps",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    results = [report(path) for path in args.paths if path.exists()]
    for result in results:
        print(render(result))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
