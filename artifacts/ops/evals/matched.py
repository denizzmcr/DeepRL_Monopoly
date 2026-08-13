"""Does more training fix the ASU-league agents? A matched-exposure test.

The first tournament ranked six policies and the ranking came out almost exactly
in order of games played, so it could not separate "the ASU league is a worse
recipe" from "1,512 games is simply early". This run separates them.

The fast league kept every snapshot, so it can be entered at the same training
volume as the ASU agents: champion_v2_h512_fast_g001512.pt has 1,512 games, and
so do v2_h512_a2_g001512 and v2_h1024_a1_g001512. Same games, same width for the
512 pair, different opponent league. Whatever separates them at that point is the
league, not the clock.

Three fast snapshots (1.5k, 15k, 47.5k) also trace the growth curve, which
answers the second half of the question -- how much a run of this shape actually
gains from 30x more games.
"""

import itertools
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

REPO = Path("/Users/denizmacbook/DeepRL_Monopoly")
sys.path.insert(0, str(REPO))

CKPT = REPO / "artifacts" / "diag" / "champion_v2"
POLICIES = {
    "fast@1.5k":  f"ppo:{CKPT}/champion_v2_h512_fast_g001512.pt",  # matched to the ASU pair
    "fast@15k":   f"ppo:{CKPT}/champion_v2_h512_fast_g015012.pt",
    "fast@47k":   f"ppo:{REPO}/artifacts/LAST_RESORT.pt",
    "asu512@1.5k": f"ppo:{CKPT}/v2_h512_a2_g001512.pt",
    "asu1024@1.5k": f"ppo:{CKPT}/v2_h1024_a1_g001512.pt",
    "asu":        "asu-value-v1",
    "kuzey":      "kuzey",
}
SEEDS = 4
WORKERS = 11
OUT = Path(__file__).with_name("matched.json")

NAMES = list(POLICIES)
TABLES = list(itertools.combinations(NAMES, 4))


def play(job):
    """One game. Returns the seating and who won, or None on truncation."""
    table, seed, rot = job
    from monopoly_game_engine.env import MonopolyEnv
    from monopoly_game_engine.train import build_opponents

    seats = [table[(i + rot) % 4] for i in range(4)]
    started = time.perf_counter()

    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=200)
    env.reset()
    agents = build_opponents([POLICIES[n] for n in seats], [0, 1, 2, 3])

    illegal = 0
    decisions = 0
    limit = env.max_rounds * 4 * 30
    while not env.done and decisions < limit:
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        allowed = env.get_allowed_actions(pid)
        action = agents[pid].choose_action(env)
        # A policy that returns an illegal action is a real defect, not a
        # rounding error -- record it rather than letting the engine absorb it.
        if action not in allowed:
            illegal += 1
            action = allowed[0] if allowed else 0
        env.step(action)
        decisions += 1

    truncated = not env.done
    win = env.winner()
    return {
        "seats": seats, "seed": seed, "rot": rot,
        "winner": None if truncated else (None if win is None else seats[win]),
        "leader": None if win is None else seats[win],
        "truncated": truncated, "rounds": env.round,
        "illegal": illegal, "secs": time.perf_counter() - started,
    }


def main():
    jobs = [(t, s, r) for t in TABLES for s in range(SEEDS) for r in range(4)]
    total = len(jobs)
    print(f"{len(NAMES)} policies, {len(TABLES)} tables, {SEEDS} seeds x 4 seats "
          f"= {total} games on {WORKERS} workers", flush=True)
    print(f"each policy plays {total * 4 // len(NAMES)} games; parity is 25%\n", flush=True)

    started = time.perf_counter()
    results = []
    with mp.Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(play, jobs, chunksize=1), 1):
            results.append(r)
            if i % 20 == 0 or i == total:
                el = time.perf_counter() - started
                eta = el / i * (total - i)
                print(f"  {i}/{total} games | {el/60:.1f} min elapsed | "
                      f"~{eta/60:.1f} min left", flush=True)
                OUT.write_text(json.dumps(results))

    OUT.write_text(json.dumps(results))
    report(results)


def report(results):
    import math

    def wilson(w, n, z=1.96):
        if not n:
            return (0.0, 0.0)
        p = w / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (100 * (c - m), 100 * (c + m))

    games = {n: 0 for n in NAMES}
    wins = {n: 0 for n in NAMES}
    leads = {n: 0 for n in NAMES}
    trunc = {n: 0 for n in NAMES}
    illegal = {n: 0 for n in NAMES}
    vs_asu = {n: [0, 0] for n in NAMES}      # games, wins in tables containing asu
    no_asu = {n: [0, 0] for n in NAMES}

    for r in results:
        pool = set(r["seats"])
        for n in pool:
            games[n] += 1
            if r["truncated"]:
                trunc[n] += 1
            bucket = vs_asu if "asu" in pool and n != "asu" else no_asu
            bucket[n][0] += 1
            if r["winner"] == n:
                bucket[n][1] += 1
        if r["winner"]:
            wins[r["winner"]] += 1
        if r["leader"]:
            leads[r["leader"]] += 1
        if r["illegal"]:
            for n in pool:
                illegal[n] += r["illegal"]

    print("\n" + "=" * 78)
    print("STANDINGS  (parity = 25%)")
    print("=" * 78)
    print(f"{'policy':10s} {'games':>6s} {'wins':>5s} {'win%':>6s} {'95% CI':>14s} "
          f"{'lead%':>6s} {'trunc%':>7s}")
    order = sorted(NAMES, key=lambda n: -wins[n] / max(games[n], 1))
    for n in order:
        lo, hi = wilson(wins[n], games[n])
        print(f"{n:10s} {games[n]:6d} {wins[n]:5d} {100*wins[n]/games[n]:6.1f} "
              f"  [{lo:5.1f},{hi:5.1f}] {100*leads[n]/games[n]:6.1f} "
              f"{100*trunc[n]/games[n]:7.1f}")

    print("\nSPLIT BY WHETHER ASU IS AT THE TABLE")
    print(f"{'policy':10s} {'with ASU':>18s} {'without ASU':>18s}")
    for n in order:
        if n == "asu":
            continue
        g1, w1 = vs_asu[n]
        g2, w2 = no_asu[n]
        a = f"{100*w1/g1:5.1f}% (n={g1})" if g1 else "     -"
        b = f"{100*w2/g2:5.1f}% (n={g2})" if g2 else "     -"
        print(f"{n:10s} {a:>18s} {b:>18s}")

    dec = sum(1 for r in results if not r["truncated"])
    print(f"\ndecided {dec}/{len(results)} games "
          f"({100*(len(results)-dec)/len(results):.1f}% hit the 200-round cap)")
    bad = sum(illegal.values())
    print(f"illegal actions: {bad if bad else 'none'}")
    print(f"results written to {OUT}")


if __name__ == "__main__":
    main()
