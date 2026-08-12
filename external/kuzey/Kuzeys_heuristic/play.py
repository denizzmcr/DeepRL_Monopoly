"""Play the champion against a panel of the organizer's own agents and print its win rate.

    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python play.py

This doubles as the package's SELF-TEST. With the defaults it must print

    champion vs mixed:  79/120 = 65.8%

If it prints anything else, this package is not the agent that was measured and nothing in
CLAUDE.md can be trusted. (The reference numbers for the other agents are in CLAUDE.md.)

THREE THINGS THIS SCRIPT DOES THAT A NAIVE HARNESS GETS WRONG:

  * SEAT ROTATION. Turn order is worth real win rate, so measuring an agent in one seat
    measures the seat as much as the agent. Games run in blocks of four with the agent in
    each seat exactly once (a Latin square) and the SAME dice across the block, so seat
    advantage cancels inside every block rather than only on average over the run.

  * PYTHONHASHSEED=0. Without it this engine is NOT reproducible: four identical runs gave
    win totals [15,15,16,15] with differing per-game results, because a container keyed by
    strings is iterated somewhere in the decision path and CPython randomises string hashing
    per process. It flips about one game in forty -- a ~2.5 pp drift, which is exactly the
    size that manufactures a fake improvement. This script re-execs itself with the seed
    pinned so you cannot forget.

  * CLAMPING THE OPPONENTS. The organizer's shipped fixed agents genuinely emit ILLEGAL
    actions: they fall through to END_TURN in the raise-funds state, where END_TURN is not
    legal, and `env.step` then raises. Unsanitised, 8 of 8 games are destroyed. They are
    clamped here exactly as upstream does it -- prefer END_TURN if legal, else the first
    legal action. OUR agent is never clamped and must never need to be; the counter below
    prints its clamp count and any non-zero value is a hard failure.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heuristic  # noqa: E402  -- binds engine/ as monopoly_game_engine; must precede engine imports
from engine.actions import ActionType  # noqa: E402
from engine.agents_fixed import TheBlocker, TheBuilder, TheGambler, TheHoarder  # noqa: E402
from engine.constants import NUM_PLAYERS  # noqa: E402
from engine.env import MonopolyEnv  # noqa: E402

END_TURN = int(ActionType.END_TURN)

#: Panels built only from agents that SHIP WITH THE ENGINE, so this file needs nothing else.
#: The full evaluation also used an ASU-class opponent, which is not part of this package --
#: see CLAUDE.md for those numbers.
PANELS: dict[str, tuple[type, ...]] = {
    "mixed": (TheBuilder, TheGambler, TheBlocker),
    "builders": (TheBuilder, TheBuilder, TheBuilder),
    "hoarders": (TheHoarder, TheHoarder, TheHoarder),
}

AGENTS = {
    "champion": lambda: heuristic.Champion(),
    "champion_plus": lambda: heuristic.ChampionPlus(),
    "spine_h100": lambda: heuristic.SpineH100(),
    "spine": lambda: heuristic.Spine(),
}


class _Opponent:
    """A shipped fixed agent, retargeted per seat and clamped to the legal set."""

    def __init__(self, cls: type) -> None:
        self._impl = cls(0)
        self.clamps = 0

    def choose_action(self, env, player_id: int, decision_seed: int) -> int:
        self._impl.player_id = int(player_id)
        legal = tuple(int(a) for a in env.get_allowed_actions(int(player_id)))
        try:
            action = int(self._impl.choose_action(env))
        except Exception:  # noqa: BLE001 -- upstream convention: fall through to END_TURN
            action = END_TURN
        if action in legal:
            return action
        self.clamps += 1
        return END_TURN if END_TURN in legal else legal[0]


class _Game:
    """The engine seeded and stepped exactly as the measurement harness does it.

    `MonopolyEnv` draws its dice from the GLOBAL `random` module, so two games running in
    one process would otherwise share and corrupt each other's stream. This captures the
    stream after construction and restores it around every step, which is what makes a
    given seed reproduce the same game every time. Copied from the harness so the numbers
    in CLAUDE.md are reproducible here.
    """

    __slots__ = ("env", "state")

    def __init__(self, seed: int, max_rounds: int) -> None:
        outer = random.getstate()
        try:
            random.seed(seed)
            self.env = MonopolyEnv(agent_ids=[0], max_rounds=max_rounds)
            self.state = random.getstate()
        finally:
            random.setstate(outer)

    def step(self, action: int):
        outer = random.getstate()
        try:
            random.setstate(self.state)
            result = self.env.step(action)
            self.state = random.getstate()
            return result
        finally:
            random.setstate(outer)


def play_one(policies, seed: int, seat: int, max_rounds: int) -> tuple[int | None, int]:
    """One game. `policies[s]` plays seat s; `seat` is where OUR agent sits."""
    game = _Game(seed, max_rounds)
    env = game.env
    our_clamps = 0
    try:
        for step in range(max_rounds * NUM_PLAYERS * 40):
            if env.done:
                break
            pid = env.whose_turn()
            legal = tuple(int(a) for a in env.get_allowed_actions(pid))
            if len(legal) == 1:
                action = legal[0]                      # forced: the policy is never consulted
            else:
                dseed = seed * 1_000_003 + step * 17 + pid
                action = int(policies[pid].choose_action(env, pid, dseed))
                if pid == seat and action not in legal:
                    our_clamps += 1                    # a hard failure; reported, never hidden
                    action = END_TURN if END_TURN in legal else legal[0]
            game.step(action)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! seed={seed} seat={seat} crashed: {type(exc).__name__}: {exc}")
        return None, our_clamps
    return (env.winner() if env.done else None), our_clamps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--agent", default="champion", choices=sorted(AGENTS))
    p.add_argument("--panel", default="mixed", choices=sorted(PANELS))
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--base-seed", type=int, default=20260811)
    p.add_argument("--max-rounds", type=int, default=200)
    a = p.parse_args(argv)

    panel = PANELS[a.panel]
    blocks = max(1, a.games // NUM_PLAYERS)
    wins = crashes = our_clamps = played = 0
    per_seat = [0] * NUM_PLAYERS

    print(f"agent={a.agent}   panel={a.panel} "
          f"({', '.join(c.__name__ for c in panel)})   games={blocks * NUM_PLAYERS}")
    print(f"base_seed={a.base_seed}   PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}   "
          f"max_rounds={a.max_rounds}")
    print("seat-rotated: each block of 4 replays the same dice with the agent in each seat once\n")

    # Agents are built ONCE and reused across games, exactly as the harness does it.
    # roster[0] is ours; 1..3 are the panel.
    roster = [AGENTS[a.agent]()] + [_Opponent(c) for c in panel]

    for block in range(blocks):
        seed = a.base_seed + block
        for rotation in range(NUM_PLAYERS):
            # The harness's cyclic Latin square: seat s is played by roster[(s + r) % 4].
            # Our agent is roster[0], so it sits wherever (s + r) % 4 == 0. Every rotation
            # in a block replays the SAME dice, which is what makes the comparison paired.
            seat_to_agent = [(s + rotation) % NUM_PLAYERS for s in range(NUM_PLAYERS)]
            policies = [roster[i] for i in seat_to_agent]
            seat = seat_to_agent.index(0)
            winner, clamped = play_one(policies, seed, seat, a.max_rounds)
            played += 1
            our_clamps += clamped
            if winner is None:
                crashes += 1
            elif winner == seat:
                wins += 1
                per_seat[seat] += 1
        if (block + 1) % 10 == 0:
            print(f"  {played:4d} games   {wins:3d} wins   {wins / played * 100:5.1f}%")

    rate = wins / max(played, 1) * 100
    print("\n" + "=" * 62)
    print(f"  {a.agent} vs {a.panel}:   {wins}/{played} = {rate:.1f}%")
    print(f"  wins by seat: {per_seat}    crashes: {crashes}    OUR clamps: {our_clamps}")
    print("=" * 62)

    if crashes or our_clamps:
        print("  !! HARD FAILURE: a crash or a clamp by our own agent invalidates this run.")
        print("     A clamp is an ILLEGAL action, and under the competition's 95% rule every")
        print("     one spends the hand-written-decision budget. Do not quote this number.")
        return 1
    if a.agent == "champion" and a.panel == "mixed" and played == 120 and a.base_seed == 20260811:
        expected = 79
        ok = wins == expected
        print(f"  SELF-TEST: expected {expected}/120 = 65.8%   ->   "
              f"{'PASS' if ok else 'FAIL -- this is not the measured agent'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
