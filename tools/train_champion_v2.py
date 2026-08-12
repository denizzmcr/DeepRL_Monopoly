"""Champion v2: built from scratch with everything two days of measurement taught us.

SETTLED BY MEASUREMENT — carried forward without re-testing
  restrict_liquidation   without it every agent scores 0%; random play goes
                         from 0/200 to 37/200 with this alone
  hybrid, 3 rules        within the 5-rule budget; the network still makes
                         ~81% of real decisions and is worth +35 points over
                         random choice in the same harness
  entropy_coef 0.005     0.05 pushes the policy toward uniform over 2,958
                         actions and costs ~30 points
  rotate_seats           deed ownership is indexed by physical seat, so a
                         pinned learner keys on position; evaluation is
                         seat-balanced
  rotating league        a fixed cast produces specialists: 86.5% on the
                         trained trio, 19% when one opponent changed
  snapshots              PPO drifts; one run peaked at 56% and ended at 26%
                         with the peak unrecoverable
  potential reward       the paper's absolute reward lost on all 3 paired
                         seeds (46.1 vs 34.1 / 35.9)

THE ONE THING NEVER TESTED, AND THE REASON THIS RUN EXISTS
  Every checkpoint we have ever trained used hidden_dim=256, the repo default.
  The paper used 1024 then 512. Learning curves across 41 checkpoints are flat
  from ~1000 games in every configuration we tried -- more games, different
  rewards, different opponents, ASU specialisation. Four interventions, four
  identical plateaus. That is the signature of a representational ceiling, not
  a training problem: the network learns everything it can express and stops.

  Two arms, 512 and 1024, trained from scratch because wider weights cannot be
  loaded from a 256-wide checkpoint. Wave 1 found fresh runs reached better
  floors than continued ones anyway.

  What to look for: if capacity is the ceiling, the wider arms keep climbing
  past game 1000 where every 256-wide run flattened. The snapshots make that
  visible directly.

LEAGUE
  ASU in 3 of 12 fields. It is the strongest opponent available and a
  competitor is building an ASU clone (reverse-engineering its rules from
  black-box probes, 73-78% agreement, targeting 90%), so ASU-like opposition is
  a known match-day condition rather than a hypothetical. A quarter is enough
  to learn from without the specialisation that made an ASU-only agent rank
  last of 11.
"""

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DIAG = REPO / "artifacts" / "diag"
OUT = DIAG / "champion_v2"
GAMES = 40000        # runs until stopped; 6 hours will not reach this
WORKERS_PER_ARM = int(__import__("os").environ.get("WORKERS", "10"))
GAMES_PER_ROUND = int(__import__("os").environ.get("ROUND", "20"))
ARMS = (512, 1024)


def league(mode="fast"):
    """Opponent fields. `mode="asu"` adds the expensive ASU fields.

    Widened for a 44-core host. On 12 cores the league was kept small because
    every ASU field costs ~55 s/game against ~1 s for the rest, and a synchronous
    collection round waits on its slowest game. With 40 workers many ASU games
    run concurrently, so breadth is affordable -- and breadth is the whole point:
    an agent trained on a narrow cast becomes a specialist, which we measured at
    86.5% on its training trio and 19.0% when one opponent changed.

    Design rules, each from a measurement:
      * no single opponent dominates -- Kuzey's heuristic is strong (77.5% vs
        Fixed-A/B/C where our champion gets 55%) and deterministic, so an agent
        seeing it too often learns its habits rather than Monopoly
      * ASU appears in several shapes, not just three-of-a-kind, because the
        realistic tournament is one strong opponent and two others
      * collapsed agents are included: the default path in this repo produces a
        0-2% agent, which is the most likely competitor submission
      * our own past agents are included: on match day every opponent is another
        team's network, and neural fields are where we score worst
    """
    champ = f"ppo:{REPO}/artifacts/CHAMPION.pt"
    nohyb = f"ppo:{DIAG}/ppo_nohybrid_mixed.pt"
    broken = f"ppo:{DIAG}/ppo_default.pt"

    asu_fields = [
        ("asu-value-v1", "asu-value-v1", "asu-value-v1"),
        ("asu-value-v1", "kuzey", "fixed-d"),
        ("asu-value-v1", champ, "fixed-d"),
        ("asu-value-v1", "fixed-b", "fixed-d"),
        ("asu-value-v1", "asu-value-v1", "kuzey"),
        ("asu-value-v1", broken, "fixed-d"),
    ]
    fields = [
        # strong heuristic opposition, essentially free at 0.07 ms/decision
        ("kuzey", "kuzey", "kuzey"),
        ("kuzey", "fixed-d", "fixed-b"),
        ("kuzey", champ, "fixed-d"),
        ("kuzey-plus", "fixed-c", "fixed-e"),
        ("kuzey", "kuzey-plus", "fixed-d"),
        ("kuzey", "fixed-e", "fixed-f"),
        # scripted breadth, weighted to the fields we score worst in
        ("fixed-a", "fixed-b", "fixed-c"),
        ("fixed-d", "fixed-e", "fixed-f"),
        ("fixed-b", "fixed-b", "fixed-b"),
        ("fixed-c", "fixed-e", "fixed-d"),
        ("fixed-b", "fixed-d", "fixed-f"),
        ("fixed-c", "fixed-c", "fixed-e"),
        ("fixed-a", "fixed-d", "fixed-f"),
        ("fixed-d", "fixed-d", "fixed-b"),
        # neural opposition: what every match-day opponent actually is
        (champ, "fixed-b", "fixed-d"),
        (champ, nohyb, "fixed-d"),
        (champ, "kuzey", "fixed-b"),
        (nohyb, "fixed-c", "fixed-e"),
        # collapsed agents: the most likely competitor submission
        (broken, "fixed-d", "fixed-b"),
        (broken, champ, "fixed-d"),
        (broken, "kuzey", "fixed-d"),
        (broken, broken, "fixed-d"),
    ]
    if mode == "asu":
        fields = fields + asu_fields
    return [f for f in fields
            if all(not x.startswith("ppo:") or Path(x[4:]).exists() for x in f)]


MODE = "fast"


def one_arm(hidden):
    sys.path.insert(0, str(REPO))
    import random

    import numpy as np
    import torch

    from monopoly_game_engine import PPOAgent
    from monopoly_game_engine.parallel_ppo import train_parallel

    seed = 1000 + hidden
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out = OUT / f"champion_v2_h{hidden}_{MODE}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    agent = PPOAgent(player_id=0, hybrid=True, device="cpu",
                     restrict_liquidation=True, entropy_coef=0.005,
                     hidden_dim=hidden)
    started = time.perf_counter()
    try:
        history = train_parallel(
            agent, n_games=GAMES, workers=WORKERS_PER_ARM, opponents=league(MODE),
            games_per_round=GAMES_PER_ROUND,   # ASU games cost ~55-80 s against ~1 s for the
                                  # rest, so ~93% of wall time sits in the 19% of
                                  # games that are ASU. With a small round most
                                  # workers finish their fast games and idle while
                                  # a couple of ASU games grind on -- measured at
                                  # 100% CPU of a possible 1200%. A large round
                                  # plus more workers than cores keeps every core
                                  # on a slow game instead of waiting for one.
            seed=seed, rotate_seats=True, log_every=500,
            checkpoint_path=str(out), checkpoint_every=500,
            keep_snapshots=True, verbose=True,
        )
        agent.save(str(out))
    except Exception as exc:
        return {"hidden": hidden, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"}
    rates = history.get("win_rates", [])
    params = sum(p.numel() for p in agent.actor.parameters())
    s = {"hidden": hidden, "status": "done", "games": GAMES,
         "actor_params": params, "seconds": round(time.perf_counter() - started, 1),
         "final": rates[-1] if rates else None,
         "best": max(rates) if rates else None, "windows": rates}
    (OUT / f"champion_v2_h{hidden}_{MODE}_summary.json").write_text(json.dumps(s, indent=2))
    return s


def main() -> int:
    # One arm per invocation. train_parallel spawns its own collection workers,
    # and a pool worker is daemonic, which cannot have children -- so the arms
    # have to be separate top-level processes rather than a nested pool.
    OUT.mkdir(parents=True, exist_ok=True)
    global MODE
    hidden = int(sys.argv[1])
    MODE = sys.argv[2] if len(sys.argv) > 2 else "fast"
    fields = league()
    asu = sum(1 for f in fields if any("asu-value" in x for x in f))
    print(f"champion v2 — hidden_dim {hidden}", flush=True)
    print(f"league: {len(fields)} fields, {asu} with ASU "
          f"({asu/len(fields)*100:.0f}% of games)", flush=True)
    started = time.perf_counter()
    r = one_arm(hidden)
    note = r.get("error") or (f"final={r.get('final')} best={r.get('best')} "
                              f"params={r.get('actor_params'):,}")
    print(f"[h{r['hidden']}] {r['status']} {note} "
          f"({(time.perf_counter()-started)/60:.0f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
