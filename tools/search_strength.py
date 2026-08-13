"""Is the search-improved teacher actually stronger than Kuzey? Measure first.

Phase 5 only pays off if this beats plain Kuzey by enough to be worth distilling.
Seat-balanced: every seed is played from all four seats, so seat advantage
cancels. Parity in a four-player game is 25%, and the opposition is three plain
Kuzeys, so anything above 25% means the search is adding something.

Run it before collecting a single label. Distilling a teacher that is not
actually better would waste the remaining time on a guaranteed dead end.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from monopoly_game_engine.actions import ActionType  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402

CFG: dict = {}


def _game(args):
    seed, seat = args
    from monopoly_game_engine.train import build_opponents
    from search_teacher import SearchKuzey

    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=200)
    env.reset()
    agents = build_opponents(["kuzey"] * 4, [0, 1, 2, 3])
    teacher = SearchKuzey(seat, playouts=CFG["playouts"],
                          horizon=CFG["horizon"], wide=CFG["wide"])
    agents[seat] = teacher

    started = time.perf_counter()
    decisions = 0
    while not env.done and decisions < 24000:
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        legal = env.get_allowed_actions(pid)
        a = agents[pid].choose_action(env)
        if a not in legal:
            a = (int(ActionType.END_TURN)
                 if int(ActionType.END_TURN) in legal else legal[0])
        env.step(a)
        decisions += 1
    return (env.winner() == seat, env.winner() is None, teacher.searches,
            teacher.free, time.perf_counter() - started)


def wilson(w, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - m), 100 * (c + m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--playouts", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--wide", type=int, default=0)
    a = ap.parse_args()
    CFG.update(playouts=a.playouts, horizon=a.horizon, wide=a.wide)

    jobs = [(1000 + s, seat) for s in range(a.seeds) for seat in range(4)]
    print(f"SearchKuzey(playouts={a.playouts}, horizon={a.horizon}, "
          f"wide={a.wide}) vs 3x kuzey | {len(jobs)} games, parity 25%",
          flush=True)

    wins = trunc = searched = free = 0
    secs = 0.0
    started = time.perf_counter()
    with mp.Pool(a.workers, initializer=CFG.update, initargs=(dict(CFG),)) as pool:
        for i, (won, tr, s, f, el) in enumerate(pool.imap_unordered(_game, jobs), 1):
            wins += int(won); trunc += int(tr); searched += s; free += f; secs += el
            if i % 20 == 0 or i == len(jobs):
                lo, hi = wilson(wins, i)
                print(f"  {i}/{len(jobs)} | win {100*wins/i:.1f}% "
                      f"[{lo:.1f},{hi:.1f}] | {secs/i:.1f}s/game", flush=True)

    lo, hi = wilson(wins, len(jobs))
    print(f"\nwin rate {100*wins/len(jobs):.1f}% [{lo:.1f}, {hi:.1f}] "
          f"over {len(jobs)} games (parity 25%)")
    print(f"searched {searched/len(jobs):.0f} decisions/game, "
          f"free {free/len(jobs):.0f}, truncated {100*trunc/len(jobs):.1f}%")
    print(f"{secs/len(jobs):.1f} s/game, wall {(time.perf_counter()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
