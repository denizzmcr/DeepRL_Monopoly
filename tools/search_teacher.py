"""A teacher stronger than Kuzey, built by adding search that only runs offline.

Distilling Kuzey caps the student at Kuzey. To pass that ceiling we need a better
teacher, and the one lever left is search: the competition forbids search *at
inference*, but nothing forbids it while generating training labels. The
submitted artefact stays a plain feed-forward network.

The method is candidate-restricted rollout. Searching all ~20 legal actions is
unaffordable, but we do not need to: Kuzey's three variants (Champion,
ChampionPlus, Spine) propose between one and three distinct actions, and they
disagree on 55% of decisions. Those disagreements are exactly the positions
where Kuzey is unsure, so the candidate set is both small and well chosen.

* All variants agree -> take it, at no cost beyond three heuristic calls.
* They disagree -> play each candidate out a few times with Kuzey driving every
  seat, truncated to a short horizon, and keep the best average.

Scoring uses the engine's own ``_compute_reward`` (net worth against the mean of
the others, clipped to +/-1), which is the same potential the RL trainer uses, so
the teacher optimises the same thing the rest of the project measures.

``deepcopy`` of a live game costs 0.14 ms, so the rollouts themselves dominate
and the horizon is the knob that matters.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monopoly_game_engine.actions import ActionType  # noqa: E402

NUM_PLAYERS = 4
VARIANTS = ("champion", "plus", "spine")


class SearchKuzey:
    """Kuzey plus truncated rollout search over the variants' proposals."""

    def __init__(self, player_id: int, playouts: int = 2, horizon: int = 8,
                 wide: int = 0):
        """``wide``: when the variants agree, still search if the legal set is
        no larger than this. Measured, the variants agree on all but ~3% of a
        seat's own decisions, so search alone changes only ~21 moves per game.
        Widening trades cost for the chance to actually improve on Kuzey rather
        than merely arbitrating between its own suggestions.
        """
        from monopoly_game_engine.train import KuzeyHeuristicOpponent

        self.player_id = player_id
        self.playouts = playouts
        self.horizon = horizon
        self.wide = wide
        self._proposers = [KuzeyHeuristicOpponent(player_id, v) for v in VARIANTS]
        # One driver per seat, reused across rollouts. Building these per
        # playout would cost more than the playout.
        self._drivers = [KuzeyHeuristicOpponent(i, "champion")
                         for i in range(NUM_PLAYERS)]
        self.searches = 0
        self.free = 0

    def _rollout(self, env, action: int) -> float:
        """Apply ``action``, then let Kuzey play everyone for a short horizon."""
        sim = copy.deepcopy(env)
        allowed = sim.get_allowed_actions(self.player_id)
        if action not in allowed:
            return -1.0
        sim.step(action)
        stop = sim.round + self.horizon
        guard = 0
        while not sim.done and sim.round < stop and guard < 400:
            guard += 1
            pid = sim.whose_turn()
            if sim.players[pid].bankrupt:
                sim._advance_turn()
                continue
            legal = sim.get_allowed_actions(pid)
            if not legal:
                legal = [int(ActionType.DO_NOTHING)]
            a = self._drivers[pid].choose_action(sim)
            if a not in legal:
                a = (int(ActionType.END_TURN)
                     if int(ActionType.END_TURN) in legal else legal[0])
            sim.step(a)
        if sim.done:
            winner = sim.winner()
            if winner is not None:
                return 1.0 if winner == self.player_id else -1.0
        if sim.players[self.player_id].bankrupt:
            return -1.0
        return float(sim._compute_reward(self.player_id))

    def choose_action(self, env) -> int:
        allowed = env.get_allowed_actions(self.player_id)
        if not allowed:
            return int(ActionType.DO_NOTHING)
        if len(allowed) == 1:
            return allowed[0]

        candidates = []
        for proposer in self._proposers:
            a = proposer.choose_action(env)
            if a in allowed and a not in candidates:
                candidates.append(a)
        if not candidates:
            return allowed[0]
        if len(candidates) == 1:
            if self.wide and len(allowed) <= self.wide:
                candidates = list(allowed)
            else:
                self.free += 1
                return candidates[0]

        self.searches += 1
        best, best_score = candidates[0], -1e9
        for a in candidates:
            score = sum(self._rollout(env, a) for _ in range(self.playouts))
            if score > best_score:
                best, best_score = a, score
        return best
