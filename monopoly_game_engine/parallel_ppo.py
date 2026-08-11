"""PPO with parallel actors: N processes collect, one process learns.

Why
---
``train.train`` plays one game at a time in one process. Against the scripted
agents nobody notices. Against ASU a game costs minutes, which is what makes
every ASU experiment unaffordable. Collection is embarrassingly parallel and the
learner is idle for almost all of it.

This is the arrangement the PPO paper describes: N actors each run the current
policy for a while, their trajectories are pooled, one update is applied, and
the new weights go back out.

What it actually buys — measured, not estimated
-----------------------------------------------
On the Windows box (Intel i3-N305, 8 physical cores at 1.8 GHz, CPU torch),
7 workers against the scripted trio:

    collection   4.30 s/game -> 0.86 s/game     5.0x
    update       1.20 s/game -> 1.10 s/game     unchanged

**Collection scales; the update does not, and it is serialised against idle
workers.** That caps the scripted case at roughly 3x however many workers you
add — Amdahl, with the update as the serial part. It does not cap the ASU case,
where one measured game was 181.4 s of collection against 1.4 s of update, so
the serial part is under 1%.

Two things eat the rest, and both are worth knowing before reading a
disappointing number off a short run:

* **Worker startup is ~20 s** (spawn plus importing torch, per worker, once).
  Irrelevant over 2,000 games, dominant over 14.
* **Stragglers set the pace of a round.** A round finishes when its slowest game
  does. Measured on ASU with ``games_per_round == workers``: 7 games took 673 s
  when the baseline game took 181 s, because one of the seven ran to the
  200-round cap. End-to-end that was 1.8x, not the ~5x collection alone
  suggests. **Set ``games_per_round`` to a multiple of ``workers`` for opponents
  whose game lengths vary** — Deal-Makers and Blockers both stall ~70% of games
  into the round cap — so a fast worker picks up another game instead of idling
  next to the straggler.

The honest summary: this makes ASU training affordable rather than free, and how
affordable depends on a setting you have to choose deliberately.

How it stays faithful to the sequential trainer
-----------------------------------------------
Workers do not reimplement the episode loop. They run ``train.run_episode``
itself, driving it with a ``_CollectingAgent`` — a real ``PPOAgent`` whose
``update()`` is a no-op and whose ``n_steps`` is unreachable. ``run_episode``
therefore does exactly what it always does (hybrid interception, potential-based
shaping, the terminal win/loss bonus, the metrics) and simply never learns
anything; the transitions pile up in the buffer and are shipped to the learner
instead. Nothing in ``train.py`` or ``agent_ppo.py`` is modified.

Two things genuinely differ from the sequential loop, both inherent to parallel
actors rather than accidental:

* **Updates land on round boundaries, not every ``n_steps`` transitions.** A
  worker collects a whole episode under one set of weights. This is more
  on-policy than the sequential loop, which updates mid-episode.
* **Advantages are normalised over the pooled round**, which is the usual
  batched-PPO behaviour and a lower-variance estimate than per-1024-step
  normalisation.

Rounds are synchronous: every game in a round is collected under the same weight
version, and results are folded into the buffer in game order regardless of
which worker finished first. So a run is reproducible from ``--seed`` and
``--games-per-round``, and no trajectory is ever stale.

Usage
-----
    python -m monopoly_game_engine.parallel_ppo --games 2000 --workers 10 \
        --opponents asu-value-v1 fixed-a fixed-b \
        --checkpoint artifacts/asu_specialist.pt --checkpoint-every 200

    # rotate through several opponent triples, one per game
    python -m monopoly_game_engine.parallel_ppo --games 4000 --workers 10 \
        --opponents fixed-a fixed-b fixed-c \
        --opponents fixed-d fixed-e fixed-f \
        --opponents asu-value-v1 fixed-b fixed-e
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .actions import ACTION_SPACE_SIZE
from .agent_ppo import PPOAgent
from .constants import NUM_PLAYERS
from .env import MonopolyEnv
from .train import DEFAULT_OPPONENT_IDS, build_opponents, normalize_opponent_plan, run_episode

# Large enough that ``run_episode`` never reaches a mid-episode update, small
# enough to stay a plain int everywhere it is compared or serialised.
_NEVER = 1 << 60

_STOP = None
_RESULT_POLL_SECONDS = 5.0


# ── Worker side ───────────────────────────────────────────────────────────────


class _CollectingAgent(PPOAgent):
    """A PPO agent that plays and records but never learns.

    ``run_episode`` calls ``update()`` at the end of every episode and, if the
    buffer is long enough, mid-episode as well. Both are suppressed here: the
    buffer is the deliverable, and clearing it in a worker would throw the
    episode away.
    """

    def update(self, last_next_state=None, last_done: bool = False) -> dict:
        return {}


def _pack_trajectory(buffer) -> dict[str, np.ndarray]:
    """Serialise a rollout for the queue.

    The per-step action masks are the bulk of it: 2958 booleans per decision,
    ~800 decisions per game. Packed to bits they are 370 bytes a step instead of
    2,958, which is the difference between ~2.3 MB and ~290 KB per game on the
    wire.
    """
    masks = torch.stack(buffer.action_masks).numpy()
    return {
        "states": np.asarray(buffer.states, dtype=np.float32),
        "actions": np.asarray(buffer.actions, dtype=np.int64),
        "log_probs": np.asarray(buffer.log_probs, dtype=np.float32),
        "rewards": np.asarray(buffer.rewards, dtype=np.float32),
        "values": np.asarray(buffer.values, dtype=np.float32),
        "dones": np.asarray(buffer.dones, dtype=np.bool_),
        "masks": np.packbits(masks, axis=1),
    }


def _restore_trajectory(agent: PPOAgent, traj: dict[str, np.ndarray]) -> int:
    """Replay a worker's rollout into the learner's buffer, step for step."""
    masks = np.unpackbits(traj["masks"], axis=1, count=ACTION_SPACE_SIZE).astype(bool)
    states = traj["states"]
    for step in range(len(states)):
        agent.store(
            states[step],
            int(traj["actions"][step]),
            float(traj["log_probs"][step]),
            float(traj["rewards"][step]),
            float(traj["values"][step]),
            bool(traj["dones"][step]),
            np.flatnonzero(masks[step]).tolist(),
        )
    return len(states)


def _worker(config: dict[str, Any], job_q, weight_q, result_q) -> None:
    """Play assigned games with the broadcast weights, return the trajectories."""
    # Each worker is one game deep; letting ten of them each open a BLAS thread
    # pool turns a 12-core machine into a scheduling fight.
    torch.set_num_threads(1)

    agent = _CollectingAgent(
        player_id=0,
        hybrid=config["hybrid"],
        hidden_dim=config["hidden_dim"],
        gamma=config["gamma"],
        lam=config["lam"],
        clip_eps=config["clip_eps"],
        entropy_coef=config["entropy_coef"],
        win_loss_bonus=config["win_loss_bonus"],
        restrict_liquidation=config["restrict_liquidation"],
        n_steps=_NEVER,
        device="cpu",
    )
    env = MonopolyEnv(agent_ids=[0], max_rounds=config["max_rounds"])
    version = -1

    while True:
        job = job_q.get()
        if job is _STOP:
            return
        index, seat, opponent_ids, wanted_version, episode_seed = job
        try:
            if version != wanted_version:
                # Every version is broadcast to every worker in order, so a
                # worker that sat out a round just discards what it missed.
                while version != wanted_version:
                    version, actor_state, critic_state = weight_q.get()
                agent.actor.load_state_dict(actor_state)
                agent.critic.load_state_dict(critic_state)

            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)

            agent.player_id = seat
            env.agent_ids = [seat]
            others = [pid for pid in range(NUM_PLAYERS) if pid != seat]
            opponents = build_opponents(opponent_ids, others)

            started = time.perf_counter()
            outcome = run_episode(env, agent, opponents, seat, is_ppo=True)
            elapsed = time.perf_counter() - started

            trajectory = _pack_trajectory(agent.buffer)
            agent.buffer.clear()
            result_q.put(
                {
                    "game": index,
                    "seat": seat,
                    "opponents": tuple(opponent_ids),
                    "won": bool(outcome["won"]),
                    "reward": float(outcome["reward"]),
                    "rounds": int(env.round),
                    "truncated": bool(env.round >= env.max_rounds),
                    "trades_initiated": int(outcome["trades_initiated"]),
                    "trades_accepted": int(outcome["trades_accepted"]),
                    "properties_acquired": int(outcome["properties_acquired"]),
                    "seconds": elapsed,
                    "trajectory": trajectory,
                }
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            import traceback

            agent.buffer.clear()
            result_q.put(
                {
                    "game": index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )


# ── Learner side ──────────────────────────────────────────────────────────────


def _snapshot(agent: PPOAgent, version: int) -> tuple:
    """Detached CPU copies of the weights.

    ``Queue.put`` pickles on a background feeder thread, so handing it the live
    tensors would let the next optimizer step rewrite them mid-pickle.
    """
    return (
        version,
        {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()},
        {k: v.detach().cpu().clone() for k, v in agent.critic.state_dict().items()},
    )


def _collect_round(
    result_q, expected: int, processes, first_round: bool = False
) -> list[dict]:
    """Wait for a whole round, failing loudly if a worker dies mid-flight."""
    results: list[dict] = []
    while len(results) < expected:
        try:
            result = result_q.get(timeout=_RESULT_POLL_SECONDS)
        except queue.Empty:
            # No worker should exit before it is told to stop. One that has
            # (segfault, OOM kill, anything outside the try below) will never
            # deliver its game, so waiting for the round to fill up is waiting
            # forever.
            dead = [
                (index, process.exitcode)
                for index, process in enumerate(processes)
                if process.exitcode is not None
            ]
            if dead:
                hint = ""
                if not results and first_round:
                    # Workers are spawned, so each child re-imports the caller's
                    # __main__. A script without an `if __name__ == "__main__":`
                    # guard therefore re-runs itself in every child and they all
                    # die on startup. It is the usual cause of "they exited
                    # before delivering anything".
                    hint = (
                        "\nNo worker delivered a single game. If train_parallel "
                        "is being called from a script, check that the call is "
                        "under `if __name__ == \"__main__\":` — spawned workers "
                        "re-import the calling module."
                    )
                raise RuntimeError(
                    f"collection worker(s) exited mid-round: {dead}; "
                    f"{len(results)}/{expected} games collected{hint}"
                )
            continue
        if "error" in result:
            raise RuntimeError(
                f"collection worker failed on game {result['game']}:\n"
                f"{result['traceback']}"
            )
        results.append(result)
    # Game order, not completion order: the learner's buffer must not depend on
    # which worker happened to finish first, or the run stops being reproducible.
    return sorted(results, key=lambda item: item["game"])


def train_parallel(
    agent: PPOAgent,
    n_games: int = 2000,
    workers: int = 4,
    games_per_round: int | None = None,
    opponents: Sequence = DEFAULT_OPPONENT_IDS,
    seed: int = 42,
    max_rounds: int = 200,
    rotate_seats: bool = True,
    log_every: int = 50,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 0,
    keep_snapshots: bool = False,
    verbose: bool = True,
) -> dict:
    """Train ``agent`` on trajectories collected by ``workers`` processes.

    ``games_per_round`` is how many games are pooled into one PPO update; it
    defaults to one game per worker. Raising it above ``workers`` costs
    freshness but smooths over stragglers, which matters when games vary from 60
    to 200 rounds. It also sets the learner's peak memory: a round is held in
    the buffer in full, at roughly 3.5 KB per decision and ~800 decisions per
    game, so ~3 MB per game pooled.

    ``rotate_seats`` defaults to True here, unlike ``train.train``: the
    observation's deed-ownership slots are indexed by physical player id, so a
    learner pinned to one seat can key on seat identity, and evaluation is
    seat-balanced.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if n_games < 1:
        raise ValueError("n_games must be at least 1")
    if games_per_round is None:
        games_per_round = workers
    if games_per_round < 1:
        raise ValueError("games_per_round must be at least 1")

    opponent_plan = normalize_opponent_plan(opponents)
    agent_pid = agent.player_id
    started_at_game = int(getattr(agent, "games_trained", 0))

    config = {
        "hybrid": agent.hybrid,
        "hidden_dim": agent.hidden_dim,
        "gamma": agent.gamma,
        "lam": agent.lam,
        "clip_eps": agent.clip_eps,
        "entropy_coef": agent.entropy_coef,
        "win_loss_bonus": agent.win_loss_bonus,
        "restrict_liquidation": getattr(agent, "restrict_liquidation", False),
        "max_rounds": max_rounds,
    }

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue()
    result_q = ctx.Queue()
    weight_qs = [ctx.Queue() for _ in range(workers)]
    processes = [
        ctx.Process(
            target=_worker,
            args=(config, job_q, weight_qs[index], result_q),
            daemon=True,
        )
        for index in range(workers)
    ]

    history: dict[str, list] = defaultdict(list)
    wins_window = 0
    games_window = 0
    truncated_total = 0
    transitions_total = 0
    wall_started = time.perf_counter()

    if verbose:
        print("=" * 64)
        print(
            f"Parallel PPO | {n_games} games | {workers} workers | "
            f"{games_per_round} games/update"
        )
        print(f"Opponent plan: {opponent_plan}")
        if started_at_game:
            print(f"Resuming from game {started_at_game}")
        print("=" * 64, flush=True)

    for process in processes:
        process.start()

    try:
        version = 0
        blob = _snapshot(agent, version)
        for weight_q in weight_qs:
            weight_q.put(blob)

        dispatched = 0
        while dispatched < n_games:
            batch = min(games_per_round, n_games - dispatched)
            for _ in range(batch):
                dispatched += 1
                absolute = started_at_game + dispatched
                seat = (
                    (agent_pid + absolute - 1) % NUM_PLAYERS
                    if rotate_seats
                    else agent_pid
                )
                job_q.put(
                    (
                        absolute,
                        seat,
                        opponent_plan[(absolute - 1) % len(opponent_plan)],
                        version,
                        seed + absolute - 1,
                    )
                )

            round_started = time.perf_counter()
            results = _collect_round(
                result_q, batch, processes, first_round=(dispatched == batch)
            )
            collect_seconds = time.perf_counter() - round_started

            update_started = time.perf_counter()
            for result in results:
                transitions_total += _restore_trajectory(agent, result["trajectory"])
            # Every trajectory ends on a terminal transition, so GAE needs no
            # bootstrap and the recursion resets at each game boundary.
            stats = agent.update(last_next_state=None, last_done=True)
            update_seconds = time.perf_counter() - update_started

            version += 1
            blob = _snapshot(agent, version)
            for weight_q in weight_qs:
                weight_q.put(blob)

            wins_window += sum(result["won"] for result in results)
            games_window += batch
            truncated_total += sum(result["truncated"] for result in results)
            agent.games_trained = started_at_game + dispatched

            if checkpoint_path and checkpoint_every > 0:
                crossed = (
                    dispatched % checkpoint_every < batch
                    or dispatched == n_games
                )
                if crossed:
                    agent.save(checkpoint_path)
                    if keep_snapshots:
                        path = Path(checkpoint_path)
                        agent.save(
                            str(
                                path.with_name(
                                    f"{path.stem}_g{agent.games_trained:06d}{path.suffix}"
                                )
                            )
                        )

            if games_window >= log_every or dispatched == n_games:
                win_rate = 100.0 * wins_window / games_window
                elapsed = time.perf_counter() - wall_started
                history["games"].append(started_at_game + dispatched)
                history["win_rates"].append(win_rate)
                history["transitions"].append(transitions_total)
                history["games_per_second"].append(dispatched / max(elapsed, 1e-9))
                history["actor_loss"].append(float(stats.get("actor_loss", 0.0)))
                history["entropy"].append(float(stats.get("entropy", 0.0)))
                if verbose:
                    print(
                        f"  Game {started_at_game + dispatched:6d} | "
                        f"Win%: {win_rate:5.1f}% | "
                        f"{dispatched / max(elapsed, 1e-9):5.2f} games/s | "
                        f"collect {collect_seconds:5.1f}s update {update_seconds:5.1f}s | "
                        f"entropy {stats.get('entropy', 0.0):.3f}",
                        flush=True,
                    )
                wins_window = 0
                games_window = 0
    finally:
        for _ in processes:
            job_q.put(_STOP)
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        for channel in (job_q, result_q, *weight_qs):
            channel.close()
            channel.cancel_join_thread()

    if checkpoint_path and checkpoint_every <= 0:
        agent.save(checkpoint_path)

    elapsed = time.perf_counter() - wall_started
    history["resumed_from_games"] = started_at_game
    history["games_completed_this_run"] = dispatched
    history["games_completed"] = int(agent.games_trained)
    history["transitions_collected"] = transitions_total
    history["truncated_games"] = truncated_total
    history["elapsed_seconds"] = elapsed
    history["games_per_second_overall"] = dispatched / max(elapsed, 1e-9)
    history["workers"] = workers
    history["games_per_round"] = games_per_round
    if verbose:
        print(
            f"\n{dispatched} games in {elapsed / 60:.1f} min "
            f"({dispatched / max(elapsed, 1e-9):.2f} games/s, "
            f"{transitions_total} transitions, "
            f"{100.0 * truncated_total / max(dispatched, 1):.1f}% hit the round cap)"
        )
    return dict(history)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO with parallel actors: N collectors, one learner",
    )
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="collection processes (default: one per core, less one for the learner)",
    )
    parser.add_argument(
        "--games-per-round",
        type=int,
        default=None,
        help=(
            "games pooled into one PPO update (default: one per worker). Use a "
            "multiple of --workers when game lengths vary, or one straggler "
            "stalls the whole round; measured 1.8x instead of ~5x on ASU"
        ),
    )
    parser.add_argument(
        "--opponents",
        nargs=3,
        action="append",
        metavar=("SEAT_1", "SEAT_2", "SEAT_3"),
        help=(
            "three opponents: fixed-a..fixed-f, asu-value-v1, asu-rollout-v1, or "
            "ppo:/path/to.pt. Repeat the flag to rotate triples, one per game."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--keep-snapshots", action="store_true")
    parser.add_argument("--resume", default=None, help="continue from a checkpoint")
    parser.add_argument("--history-out", default=None, help="write history JSON here")
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.005,
        help="0.005 is the settled value; the 0.05 default collapsed to 2%%",
    )
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    parser.add_argument(
        "--no-restrict-liquidation",
        dest="restrict_liquidation",
        action="store_false",
        help="allow voluntary mortgage/sell; measured at 0%% win rate, for ablations only",
    )
    parser.add_argument("--no-rotate-seats", dest="rotate_seats", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    opponents = args.opponents if args.opponents else [list(DEFAULT_OPPONENT_IDS)]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = PPOAgent(
        player_id=args.player_id,
        hybrid=args.hybrid,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        restrict_liquidation=args.restrict_liquidation,
        device="cpu",
    )
    if args.resume:
        # load() also restores the checkpoint's own training_config, so a resumed
        # run keeps the hyperparameters it was trained with.
        agent.load(args.resume)

    history = train_parallel(
        agent,
        n_games=args.games,
        workers=args.workers,
        games_per_round=args.games_per_round,
        opponents=opponents,
        seed=args.seed,
        max_rounds=args.max_rounds,
        rotate_seats=args.rotate_seats,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        keep_snapshots=args.keep_snapshots,
    )

    if args.history_out:
        target = Path(args.history_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(history, indent=2, default=float), encoding="utf-8")
        print(f"History written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["train_parallel", "main"]
