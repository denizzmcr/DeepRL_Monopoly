"""Where every policy we have actually stands, measured against each other.

Not a training run. Nothing here learns; every policy is frozen and plays
deterministically.

The engine seats four players, and we have six policies, so "round robin" means
every 4-subset of the six -- C(6,4) = 15 tables. Each table is played over
SEEDS seeds x 4 cyclic seat rotations, so within a table every policy sits in
every seat equally often and all four see identical dice. Seat matters here:
deed ownership in the observation is indexed by physical player id, so a policy
can key on seat identity, which is why training rotates seats too.

Each policy therefore appears in C(5,3) = 10 of the 15 tables and plays
10 x SEEDS x 4 games. Parity in a four-player game is 25%.

Cost is dominated by ASU: ~55 s/game against ~1 s for everything else, and ASU
is present in 10 of the 15 tables. That is the whole wall clock.

Truncation is reported, not hidden. A game that hits the 200-round cap has no
winner, and past measurements found fields where 73% of games ended that way --
a policy can look weak simply for playing in a stalemate field, so decided
games and truncations are counted separately.
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
D = REPO / "artifacts" / "distill"
POLICIES = {
    "d_256":    f"ppo:{D}/student2_c3_h256.pt",
    "d_512a":   f"ppo:{D}/student2_c1_h512.pt",
    "d_512b":   f"ppo:{D}/student2_c4_h512.pt",
    "d_1024":   f"ppo:{D}/student2_c2_h1024.pt",
    "fast":     f"ppo:{REPO}/artifacts/LAST_RESORT.pt",
    "asu":      "asu-value-v1",
    "kuzey":    "kuzey",
}
SEEDS = 4
WORKERS = 11
OUT = Path(__file__).with_name("finals.json")

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
