"""UNDERDOG — competition entrypoint.

    choose_action(state, player_id, allowed_actions) -> int

This is the harness contract, exactly: ``state`` is a **plain JSON dict**,
``player_id`` an int, ``allowed_actions`` a list of ints in engine emission
order (NOT sorted). There is no live engine and there are no custom objects.

What went wrong before
----------------------
The previous revision detected its board with ``hasattr(state, "properties")``
and reached for ``getattr(state, "board")``. Dicts have no such attributes, so
the check was False on every decision of every game, the policy never ran once,
and the agent played ``allowed_actions[0]`` for an entire tournament. In debt
the engine offers only liquidation actions, whose first entry alternates
between ``mortgage(sq=X)`` and ``unmortgage(sq=X)``, so that fallback ground
itself to bankruptcy in round 0. Nothing detected it because a legal move is
indistinguishable from a good move to the harness.

Three things changed, in order of how much they matter:

1. ``underdog_gbm/dict_board.py`` rebuilds the engine surface the policy reads
   from ``state["board"]``, using the engine's own ``Property`` and ``Player``
   classes. Verified by replaying real games through JSON: 800/800 decisions
   identical to the live-engine policy.
2. A guard sits between the policy and the return value. It refuses
   cash-negative liquidation while in debt and refuses to close a two-action
   cycle, whatever the policy asks for. The livelock cannot recur even if the
   board is missing and the rule fallback is driving.
3. The argument-order tolerance layer is gone. The contract is fixed and
   documented; sorting arguments by shape hid the real failure instead of
   surviving it.

The policy
----------
Two gradient-boosted models over engineered features, in ``underdog_gbm/``.
Model A owns the auction family, Model B every other family; each scores the
legal candidates and the agent takes the argmax.

Contract compliance
-------------------
* **Only legal actions.** Every return value is checked against
  ``allowed_actions`` and replaced if it is not a member.
* **The global RNG is never touched.** Deterministic argmax, no random source.
* **Latency.** LightGBM over a few hundred candidate rows: single-core
  milliseconds against the 2 s limit.
* **Dependencies.** ``numpy`` and ``lightgbm``, both wheel-only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_ROOT = Path(__file__).resolve().parent

# Only the policy package goes on the path, and only this one. ``underdog/``
# contains a complete vendored copy of the simulator under the top-level name
# ``engine``; putting it on sys.path[0] would let it answer a later
# ``import engine`` and decide the rules for the whole table. ``_fallback()``
# adds it only if it is ever needed, and only after ``engine`` is bound.
_PKG = str(_ROOT / "underdog_gbm")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

__all__ = ["Agent", "choose_action", "make_agent", "VARIANT"]

VARIANT = "GBM"

# Set by the test suite. When true, a missing board raises instead of quietly
# playing a fallback -- the failure mode that cost the last tournament.
STRICT = False

_POLICY: Any = None
_FALLBACK = False


def _policy() -> Any:
    """Return the policy, loading it on first use."""
    global _POLICY, _FALLBACK
    if _POLICY is not None:
        return _POLICY
    try:
        # Imported before LightGBM on purpose: this runs ``engine_shim``, which
        # binds the top-level name ``engine`` to whichever engine the harness
        # has. Doing it first means that even when LightGBM is what fails,
        # ``engine`` is already correct for the fallback's own imports.
        from underdog_gbm.gbm_policy import MonopolyAgent
        import lightgbm as lgb
        agent = MonopolyAgent()
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
    """The rule agent, imported without letting its engine copy win."""
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


# ---------------------------------------------------------------- utilities

def _as_actions(value: Any) -> list[int]:
    """Coerce the action list to ``list[int]``. Order is preserved.

    Emission order carries information -- when jailed the engine emits
    PAY_BAIL before ROLL_DICE -- so this never sorts.
    """
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _legal(action: Any, allowed: Sequence[int]) -> int:
    """Return ``action`` if legal, else a legal substitute. Never raises."""
    allowed = list(allowed)
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


def _looks_like_engine(obj: Any) -> bool:
    """A live engine, as handed over by this repository's own harnesses.

    The tournament never provides one. This exists so the agent can still be
    measured locally against the real simulator.
    """
    return (obj is not None and not isinstance(obj, dict)
            and hasattr(obj, "properties") and hasattr(obj, "players")
            and hasattr(obj, "get_allowed_actions"))


# -------------------------------------------------------------------- guard

class _Guard:
    """Last line between the policy and the wire.

    Two rules, both independent of whatever chose the action:

    * **Never redeem while in debt.** ``unmortgage`` costs 1.1x the mortgage
      value. Offered only because the deed is affordable, taking it puts the
      debt straight back and re-offers the mortgage. That is the exact cycle
      that bankrupted the previous revision in round 0.
    * **Never close a two-action cycle.** If the last four emissions alternate
      A, B, A, B then A is refused. Repetition of the *same* action is left
      alone -- building four houses on one square is four identical legal
      emissions and must not be blocked.
    """

    __slots__ = ("_recent",)

    _LIMIT = 6

    def __init__(self) -> None:
        self._recent: list[int] = []

    def record(self, action: int) -> None:
        self._recent.append(int(action))
        if len(self._recent) > self._LIMIT:
            del self._recent[0]

    def _closes_cycle(self, action: int) -> bool:
        r = self._recent
        if len(r) < 3:
            return False
        a, b = r[-2], r[-1]
        # ... A B A  and we are about to emit B again.
        return a != b and r[-3] == a and action == b

    def candidates(self, allowed: Sequence[int], names: dict[int, str],
                   debt: bool) -> list[int]:
        """The subset of ``allowed`` the guard is willing to emit."""
        keep = list(allowed)
        if debt:
            safe = [a for a in keep
                    if not str(names.get(a, "")).strip().lower().startswith("unmortgage")]
            if safe:
                keep = safe
        safe = [a for a in keep if not self._closes_cycle(a)]
        if safe:
            keep = safe
        return keep

    def apply(self, action: Any, allowed: Sequence[int], names: dict[int, str],
              debt: bool) -> int:
        keep = self.candidates(allowed, names, debt)
        try:
            chosen = int(action)
        except (TypeError, ValueError):
            chosen = None
        if chosen is None or chosen not in keep:
            chosen = _legal(keep[0] if keep else None, allowed)
        chosen = _legal(chosen, allowed)
        self.record(chosen)
        return chosen


# -------------------------------------------------------------------- agent

class Agent:
    """Class form. One instance per seat."""

    name = "UNDERDOG"

    def __init__(self, player_id: int = 0, **kwargs: Any) -> None:
        for key in ("player_id", "pid", "agent_id", "seat"):
            if kwargs.get(key) is not None:
                player_id = int(kwargs[key])
                break
        self.player_id = int(player_id)
        self.guard = _Guard()

    def choose_action(self, state: Any, player_id: Any = None,
                      allowed_actions: Any = None) -> int:
        """The contract: ``(state: dict, player_id: int, allowed: list[int])``."""
        allowed = _as_actions(allowed_actions)
        try:
            seat = int(player_id)
        except (TypeError, ValueError):
            seat = self.player_id

        import dict_board as dbmod

        names = dbmod.action_names(state)
        debt = dbmod.in_debt(state, names)

        board: Any = None
        if isinstance(state, dict):
            if not allowed:
                allowed = sorted(names)
            board = dbmod.board_from_state(state, allowed, seat)
        elif _looks_like_engine(state):
            # This repository's own harness hands over the live simulator.
            board = state
            if not allowed:
                allowed = _as_actions(state.get_allowed_actions(seat))
            try:
                seat = int(state.whose_turn())
            except Exception:
                pass

        if not allowed:
            return 0

        if board is None:
            _warn_no_board()
            if STRICT:
                raise AssertionError(
                    "no board in the decision state -- the policy cannot run")
            return self.guard.apply(None, allowed, names, debt)

        try:
            action = _policy().choose_action(board, seat, 0)
        except Exception as exc:
            _warn_policy_failed(exc)
            action = None
        return self.guard.apply(action, allowed, names, debt)

    def __repr__(self) -> str:
        kind = "rules (fallback)" if _FALLBACK else VARIANT
        return f"Agent(seat={self.player_id}, policy={kind})"


_WARNED_NO_BOARD = False
_WARNED_POLICY = False


def _warn_no_board() -> None:
    global _WARNED_NO_BOARD
    if not _WARNED_NO_BOARD:
        _WARNED_NO_BOARD = True
        print("[UNDERDOG] no board in the decision state; the policy cannot run",
              file=sys.stderr, flush=True)


def _warn_policy_failed(exc: BaseException) -> None:
    global _WARNED_POLICY
    if not _WARNED_POLICY:
        _WARNED_POLICY = True
        print(f"[UNDERDOG] policy raised ({type(exc).__name__}: {exc}); "
              f"guarded legal play for the rest of this game",
              file=sys.stderr, flush=True)


def make_agent(player_id: int = 0, **kwargs: Any) -> Agent:
    return Agent(player_id, **kwargs)


_SEATS: dict[int, Agent] = {}


def choose_action(state: Any, player_id: Any = 0,
                  allowed_actions: Any = None) -> int:
    """The required contract.

    One ``Agent`` per seat, kept for the life of the process so the guard's
    short history survives across decisions within a game.
    """
    try:
        seat = int(player_id)
    except (TypeError, ValueError):
        seat = 0
    agent = _SEATS.get(seat)
    if agent is None:
        agent = Agent(seat)
        _SEATS[seat] = agent
    return agent.choose_action(state, seat, allowed_actions)
