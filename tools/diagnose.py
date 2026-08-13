"""Why does UNDERDOG lose in a mixed field? Record what the win rate hides.

``gauntlet.py`` records who won, which is the result but not the cause. A 18.8%
win rate is consistent with two opposite failures -- going bankrupt too often,
or surviving comfortably while accumulating too little -- and the fix for one
is the opposite of the fix for the other. This replays the same round-robin
tables and records the end state and the behaviour of every seat, so the two
can be told apart.

Per seat, per game: whether it won, whether it went bankrupt and when, its
final net worth and its rank, what it held, and how it spent its decisions
across the action families. Everything is written per game, so any breakdown
can be computed afterwards without replaying.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gauntlet as G  # noqa: E402

NUM_PLAYERS = 4
MAX_ROUNDS = 200


def _family(action: int) -> str:
    """Which action family an index belongs to, by the engine's own offsets."""
    from monopoly_game_engine.actions import OFFSETS
    best, name = -1, "binary"
    for key, start in OFFSETS.items():
        if action >= start and start >= best:
            best, name = start, key
    return name


def _snapshot(env, pid: int) -> dict:
    p = env.players[pid]
    return {
        "net_worth": float(p.net_worth()),
        "cash": float(p.cash),
        "deeds": len(p.properties),
        "monopolies": int(p.num_monopolies()),
        "houses": int(sum(getattr(q, "houses", 0) for q in p.properties)),
        "mortgaged": int(sum(1 for q in p.properties if q.mortgaged)),
        "bankrupt": bool(p.bankrupt),
    }


def _play(lineup: tuple[str, ...], seed: int) -> dict:
    import random
    from collections import Counter
    from monopoly_game_engine.env import MonopolyEnv

    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=MAX_ROUNDS)
    env.reset()
    seats = [G._seat(n, i) for i, n in enumerate(lineup)]

    fams = [Counter() for _ in range(NUM_PLAYERS)]
    bankrupt_round = [None] * NUM_PLAYERS
    # Net worth every 20 rounds: a curve separates "never got going" from
    # "was ahead and lost it", which the final number alone cannot.
    curve = [[] for _ in range(NUM_PLAYERS)]
    next_mark = 20

    decisions = 0
    while not env.done and decisions < 24000:
        pid = env.whose_turn()
        for i in range(NUM_PLAYERS):
            if bankrupt_round[i] is None and env.players[i].bankrupt:
                bankrupt_round[i] = env.round
        if env.round >= next_mark:
            for i in range(NUM_PLAYERS):
                curve[i].append(round(float(env.players[i].net_worth())))
            next_mark += 20
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        legal = env.get_allowed_actions(pid)
        if not legal:
            env._advance_turn()
            continue
        action = seats[pid].choose_action(env, legal)
        fams[pid][_family(action)] += 1
        # Record how often a family was *available* as well as chosen: a family
        # never used may mean the policy declines it or never sees it.
        env.step(action)
        decisions += 1

    for i in range(NUM_PLAYERS):
        if bankrupt_round[i] is None and env.players[i].bankrupt:
            bankrupt_round[i] = env.round

    winner = env.winner()
    finals = [_snapshot(env, i) for i in range(NUM_PLAYERS)]
    order = sorted(range(NUM_PLAYERS), key=lambda i: -finals[i]["net_worth"])
    rank = {pid: r + 1 for r, pid in enumerate(order)}

    return {
        "seed": seed,
        "rounds": int(env.round),
        "truncated": env.round >= MAX_ROUNDS,
        "winner": lineup[winner] if winner is not None else None,
        "seats": [
            {
                "name": lineup[i],
                "seat": i,
                "won": winner == i,
                "rank": rank[i],
                "bankrupt_round": bankrupt_round[i],
                "families": dict(fams[i]),
                "decisions": int(sum(fams[i].values())),
                "curve": curve[i],
                **finals[i],
            }
            for i in range(NUM_PLAYERS)
        ],
    }


def _job(args):
    lineup, seed = args
    try:
        return _play(tuple(lineup), seed)
    except Exception as exc:
        return {"seed": seed, "fatal": f"{type(exc).__name__}: {exc}",
                "seats": [], "winner": None, "truncated": False, "rounds": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--candidate", default="UNDERDOG")
    ap.add_argument("--exclude", nargs="*", default=["boom"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rivals = [r for r in G.RIVALS if r not in (a.exclude or ())]
    cost = {"6c0de": 68.0, "inncenta": 44.0, "expo": 3.1,
            "boom": 0.5, "aline": 0.2, "slayer": 0.2}
    rivals.sort(key=lambda r: -cost.get(r, 1.0))

    jobs = []
    for combo in itertools.combinations(rivals, 3):
        table = [a.candidate, *combo]
        for s in range(a.seeds):
            for rot in range(NUM_PLAYERS):
                jobs.append(([table[(i - rot) % 4] for i in range(4)], 7000 + s))

    print(f"diagnosing {a.candidate}: {len(jobs)} games", flush=True)
    out = []
    started = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=G._pin_engine) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            out.append(r)
            if i % 40 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  "
                      f"{(time.perf_counter()-started)/60:.1f} min", flush=True)

    path = Path(a.out) if a.out else (REPO / "artifacts" /
                                      f"diag_{a.candidate}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    G._pin_engine()
    main()
