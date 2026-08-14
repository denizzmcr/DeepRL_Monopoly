"""UNDERDOG — competition entrypoint.

    choose_action(state, allowed_actions, env, player_id) -> int

``env`` and ``player_id`` are the optional extras the harness passes by keyword
when they are declared after the two required parameters. This agent declares
them because it needs the board itself: its features read deeds, colour groups,
houses, cash, the house bank and pending trades, and the 300-float state vector
cannot be turned back into an environment.

Both the function form and the class form are exported, so either calling
convention works. A module-level ``Agent`` keeps one policy object per seat.

The policy
----------
Two gradient-boosted models over engineered features, in ``underdog_gbm/``.
Model A owns the auction family, Model B every other family; each scores the
legal candidates and the agent takes the argmax. See
``underdog_gbm/gbm_policy.py`` for the split and ``docs/GAUNTLET.md`` for what
it measured against the other teams' agents.

Contract compliance
-------------------
* **Only legal actions.** Every return value is checked against
  ``allowed_actions`` and replaced if it is not a member. The rest of this
  repository prefers to fail closed on an illegal action, which is right for
  development because it surfaces bugs — but in a scored match an exception and
  an illegal action both lose the game, so here it substitutes instead.
* **The global RNG is never touched.** The policy is a deterministic argmax and
  draws from no random source, global or private. Nothing here calls
  ``random.*``, ``np.random.*`` or ``torch.rand*``.
* **Latency.** LightGBM inference over a few hundred candidate rows: single-core
  milliseconds, against the per-decision limit.
* **Dependencies.** ``numpy`` and ``lightgbm``, both wheel-only, both in
  ``requirements.txt``. Nothing is downloaded at match time.

The fallback, and why it is here
--------------------------------
LightGBM is a compiled extension and the boosters are files on disk. If either
fails to load — a wheel that will not import in the sandbox, a missing OpenMP
runtime, a truncated checkout — this falls back to ``ChampionScore``, the
hand-written rule agent in ``underdog/``, which needs no dependency beyond the
engine. That is a real downgrade and it is announced on ``stderr`` once rather
than absorbed silently, because a submission that quietly plays a weaker policy
than the one that was measured is worse than one that says so. It is preferred
to the alternative: without it, an import failure forfeits every game.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parent

# Only the policy package goes on the path, and only this one.
#
# ``underdog/`` is deliberately NOT added here even though the fallback lives
# in it: it contains ``underdog/engine/``, a complete vendored copy of the
# simulator under the top-level name ``engine``. Putting that on sys.path[0]
# inside the harness's process would let it answer a later ``import engine``
# and silently decide the rules for the whole table -- the exact failure this
# project pinned its own engine to avoid (docs/GAUNTLET.md). ``_fallback()``
# adds it only if it is ever needed, and only after ``engine`` is already
# bound to the harness's simulator.
_PKG = str(_ROOT / "underdog_gbm")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

__all__ = ["Agent", "choose_action", "make_agent", "VARIANT"]

VARIANT = "GBM"

# Resolved lazily so importing this module cannot fail on a missing package.
_POLICY: Any = None
_FALLBACK = False


def _policy() -> Any:
    """Return the policy, loading it on first use.

    The boosters are ~4.3 MB of text and LightGBM is a compiled extension, so
    neither is touched at import time. A failure here is reported once and then
    served by the rule agent for the rest of the process.
    """
    global _POLICY, _FALLBACK
    if _POLICY is not None:
        return _POLICY
    try:
        # Imported before LightGBM on purpose. This module runs ``engine_shim``,
        # which binds the top-level name ``engine`` to whichever
        # ``monopoly_game_engine`` the harness already has. Doing it first means
        # that even when LightGBM is what fails, ``engine`` is correctly bound
        # before the fallback -- whose own imports would otherwise resolve
        # against ``underdog/engine/`` and run on the wrong copy of the rules.
        from underdog_gbm.gbm_policy import MonopolyAgent
        import lightgbm as lgb
        agent = MonopolyAgent()
        # Load both boosters now. They are lazy by default, and a truncated or
        # unreadable model file should surface here rather than mid-game on the
        # first decision of its family -- Model A's family is auctions, which a
        # game can reach hundreds of decisions in.
        for booster in (agent.model_a, agent.model_b):
            booster._b = lgb.Booster(model_file=booster.path)
        _POLICY = agent
    except Exception as exc:
        print(f"[UNDERDOG] gradient-boosted policy unavailable ({type(exc).__name__}:"
              f" {exc}); falling back to the rule agent", file=sys.stderr, flush=True)
        _POLICY = _fallback()
        _FALLBACK = True
    return _POLICY


def _fallback() -> Any:
    """The rule agent, imported without letting its engine copy win.

    ``underdog/heuristic`` reads ``from engine.actions import ...``. The name
    ``engine`` must already point at the harness's simulator before that import
    runs, or Python will bind it to ``underdog/engine/`` -- a vendored copy --
    and the fallback would then play by rules the rest of the table is not
    using. So bind it explicitly first, and refuse to import at all if there is
    no simulator to bind it to.
    """
    import importlib
    if "engine" not in sys.modules:
        engine = importlib.import_module("monopoly_game_engine")
        sys.modules["engine"] = engine
        for sub in ("actions", "constants", "env", "state"):
            sys.modules.setdefault(
                f"engine.{sub}", importlib.import_module(f"monopoly_game_engine.{sub}"))
    path = str(_ROOT / "underdog")
    if path not in sys.path:
        sys.path.append(path)      # appended, never ahead of the harness
    from heuristic import ChampionScore
    return ChampionScore()


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
        kind = "rules (fallback)" if _FALLBACK else VARIANT
        return f"Agent(seat={self.player_id}, policy={kind})"


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
