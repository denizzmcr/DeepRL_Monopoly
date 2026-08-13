"""UNDERDOG — competition entrypoint.

    choose_action(state, allowed_actions, env, player_id) -> int

``env`` and ``player_id`` are the optional extras the harness passes by
keyword when they are declared after the two required parameters. This agent
declares them because it needs the board itself: the policy is a set of
hand-written rules that read deeds, colour groups, houses, cash and pending
trades, and the 300-float state vector cannot be turned back into an
environment.

Both the function form and the class form are exported, so either calling
convention works. A module-level ``Agent`` keeps one policy object per seat.

Contract compliance
-------------------
* **Only legal actions.** Every return value is checked against
  ``allowed_actions`` and replaced if it is not a member. The rest of this
  repository prefers to fail closed on an illegal action, which is right for
  development because it surfaces bugs — but in a scored match an exception
  and an illegal action both lose the game, so here it substitutes instead.
* **The global RNG is never touched.** The policy is deterministic; its only
  stochastic branch draws from a private ``random.Random(decision_seed)`` and
  is disabled at ``EPSILON = 0``. Nothing here calls ``random.*``,
  ``np.random.*`` or ``torch.rand*``.
* **Latency.** Rules only, no search and no network: microseconds per
  decision against the 5 s limit.

The policy is ``ChampionScore`` — see ``underdog/heuristic/champion.py`` and
``docs/GAUNTLET.md`` for what it is and what it measured.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parent
_HEURISTIC = _ROOT / "underdog"
if str(_HEURISTIC) not in sys.path:
    sys.path.insert(0, str(_HEURISTIC))

__all__ = ["Agent", "choose_action", "make_agent", "VARIANT"]

VARIANT = "ChampionScore"

# Resolved lazily so importing this module cannot fail on a missing package.
_POLICY: Any = None


def _policy() -> Any:
    global _POLICY
    if _POLICY is None:
        from heuristic import ChampionScore
        _POLICY = ChampionScore()
    return _POLICY


def _legal(action: Any, allowed: Sequence[int]) -> int:
    """Return ``action`` if it is legal, else a legal substitute.

    Never raises and never returns a non-member of ``allowed``: an illegal
    return fails the match outright, so there is no value in propagating.
    """
    allowed = [int(a) for a in allowed]
    if not allowed:
        return 0
    try:
        action = int(action)
    except (TypeError, ValueError):
        return allowed[0]
    if action in allowed:
        return action
    try:
        from monopoly_game_engine.actions import ActionType
        end = int(ActionType.END_TURN)
        if end in allowed:
            return end
    except Exception:
        pass
    return allowed[0]


class Agent:
    """Class form. One instance per seat; holds no state between games."""

    name = "UNDERDOG"

    def __init__(self, player_id: int = 0, **kwargs: Any) -> None:
        for key in ("player_id", "pid", "agent_id", "seat"):
            if kwargs.get(key) is not None:
                player_id = int(kwargs[key])
                break
        self.player_id = int(player_id)

    def choose_action(self, state=None, allowed_actions=None, env=None,
                      player_id=None) -> int:
        seat = self.player_id if player_id is None else int(player_id)
        if env is None:
            # No board: nothing this policy reads is recoverable from the
            # state vector, so take a legal action rather than guess.
            return _legal(None, allowed_actions or [])
        # The engine only asks an agent to act on its own turn, so trust the
        # engine's view of the acting seat over the one passed at construction.
        try:
            seat = int(env.whose_turn())
        except Exception:
            pass
        if allowed_actions is None:
            try:
                allowed_actions = list(env.get_allowed_actions(seat))
            except Exception:
                allowed_actions = []
        try:
            action = _policy().choose_action(env, seat, 0)
        except Exception:
            action = None
        return _legal(action, allowed_actions)

    def __repr__(self) -> str:
        return f"Agent(seat={self.player_id}, variant={VARIANT})"


def make_agent(player_id: int = 0, **kwargs: Any) -> Agent:
    return Agent(player_id, **kwargs)


_SEATS: dict[int, Agent] = {}


def choose_action(state, allowed_actions, env=None, player_id=None) -> int:
    """The required contract, with the two optional extras declared."""
    seat = 0 if player_id is None else int(player_id)
    agent = _SEATS.get(seat)
    if agent is None:
        agent = Agent(seat)
        _SEATS[seat] = agent
    return agent.choose_action(state, allowed_actions, env, seat)
