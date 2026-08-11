"""The submitted agent, and nothing else.

This is the only file the match harness needs to import. It loads a trained PPO
checkpoint and turns it into a policy: ``SubmissionAgent(seat).choose_action(env)``
returns one legal action index.

What this module deliberately does *not* do
-------------------------------------------
**It never imports ``ASU_FROZEN_TEACHER``.** ASU is allowed as a training
opponent and as a benchmark, and nothing else; no ASU code may run at match
time, and nothing the submission imports may reach it. The inference path here
is therefore written out locally rather than reusing
``ASU_FROZEN_TEACHER.evaluate._NeuralAdapter``, which is otherwise the same
algorithm. ``tests/test_submission_agent.py`` imports this module in a fresh
interpreter and fails if ``ASU_FROZEN_TEACHER`` shows up in ``sys.modules``, and
separately asserts that the two paths pick identical actions, so the measured
results carry over.

It also never falls back to a different action when its choice is rejected. A
policy that silently substitutes ``END_TURN`` for an illegal choice looks like it
is working while playing a policy nobody measured. Here an illegal choice raises
``IllegalActionError``.

The three hardcoded rules
------------------------
1. ``fixed_buy_decision``          — buy if it completes a monopoly, else if $100 would remain
2. ``fixed_accept_trade_decision`` — accept if it completes a monopoly, else if net worth change >= 0
3. ``restrict_actions``            — never voluntarily mortgage or sell

Rules 1 and 2 are the hybrid split the network was trained under: those two
decisions were never routed through the actor, so they have to stay outside it
at match time too.

Rule 3 is the one that decides the game. Voluntary liquidation was masked out
during training, so those logits never received a gradient and are still at
initialisation. The same checkpoint run *without* the restriction scores 0% and
liquidates 251 times per game. A checkpoint trained with the restriction is
wrong to run without it, which is why the flag travels inside the checkpoint
(``training_config.restrict_liquidation``) and is read from there rather than
passed in. Checkpoints predating the flag default to unrestricted, which is how
they were trained.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:  # importable from outside the repo root
    sys.path.insert(0, str(_ROOT))

from monopoly_game_engine.action_filters import restrict_actions  # noqa: E402
from monopoly_game_engine.actions import (  # noqa: E402
    ACTION_SPACE_SIZE,
    ActionType,
    action_to_description,
)
from monopoly_game_engine.agent_ppo import (  # noqa: E402
    fixed_accept_trade_decision,
    fixed_buy_decision,
)
from monopoly_game_engine.constants import RULESET_VERSION  # noqa: E402
from monopoly_game_engine.networks import ActorNetwork  # noqa: E402
from monopoly_game_engine.state import STATE_DIM  # noqa: E402

DEFAULT_CHECKPOINT = _ROOT / "artifacts" / "CHAMPION.pt"
CHECKPOINT_FORMAT_VERSION = 3

_BUY = int(ActionType.BUY_PROPERTY)
_ACCEPT = int(ActionType.ACCEPT_TRADE)
_DECLINE = int(ActionType.DECLINE_TRADE)
_HYBRID_FIXED = frozenset((_BUY, _ACCEPT))


class IllegalActionError(RuntimeError):
    """The agent was about to return an action the engine did not offer.

    Raised instead of substituting a legal action, so a policy bug surfaces as a
    crash in testing rather than as quietly degraded play in a match.
    """


class CheckpointError(ValueError):
    """The checkpoint is missing, unreadable, or not this ruleset."""


def checkpoint_sha256(path: str | Path) -> str:
    """SHA-256 of a checkpoint file, for recording exactly what was submitted."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_actor(path: Path) -> tuple[ActorNetwork, dict[str, Any]]:
    """Load the actor weights, refusing anything that is not this ruleset.

    Only the actor is loaded. The critic and the optimizer state are training
    machinery; at match time they are dead weight (about two thirds of the
    file), and loading them would only add ways to fail.
    """
    if not path.is_file():
        raise CheckpointError(f"Missing checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised as one checkpoint error
        raise CheckpointError(f"Unable to load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint {path} is not a mapping")

    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "ruleset": RULESET_VERSION,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_SPACE_SIZE,
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise CheckpointError(
            f"Incompatible checkpoint metadata at {path}: {actual}; expected {expected}"
        )
    for key in ("hybrid", "hidden_dim", "actor"):
        if key not in payload:
            raise CheckpointError(f"Checkpoint {path} is missing {key!r}")

    actor = ActorNetwork(int(payload["hidden_dim"]))
    try:
        actor.load_state_dict(payload["actor"])
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint {path} does not fit this ActorNetwork: {exc}"
        ) from exc
    actor.eval()
    actor.requires_grad_(False)

    training_config = payload.get("training_config") or {}
    metadata = {
        "hybrid": bool(payload["hybrid"]),
        "hidden_dim": int(payload["hidden_dim"]),
        # Absent on checkpoints predating the flag; those were trained without
        # the restriction, so False reproduces their training conditions.
        "restrict_liquidation": bool(training_config.get("restrict_liquidation", False)),
        "games_trained": int(payload.get("games_trained", 0)),
        "step_count": int(payload.get("step_count", 0)),
        "trained_as_seat": payload.get("player_id"),
        "ruleset": payload["ruleset"],
    }
    return actor, metadata


class SubmissionAgent:
    """A trained PPO checkpoint playing one seat, deterministically, on CPU.

    ``player_id`` is the seat this instance plays, which need not be the seat
    the checkpoint trained in: the observation is built per-player and training
    rotated the learner through all four seats, so the weights are seat-agnostic.

    Deterministic on purpose. Training sampled from the policy to explore;
    a match wants its best guess, so this takes the argmax over the legal
    actions. Ties break to the lowest action index.
    """

    def __init__(self, player_id: int, checkpoint: str | Path = DEFAULT_CHECKPOINT):
        self.player_id = int(player_id)
        self.checkpoint = Path(checkpoint)
        self.actor, metadata = _load_actor(self.checkpoint)
        self.hybrid: bool = metadata["hybrid"]
        self.restrict_liquidation: bool = metadata["restrict_liquidation"]
        self.metadata = metadata

        # Counters, so a match log can show how often each rule actually fired.
        self.decisions = 0
        self.hybrid_decisions = 0
        self.restricted_decisions = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def choose_action(self, env) -> int:
        """Return one legal action index for this seat in the current state."""
        legal = list(env.get_allowed_actions(self.player_id))
        if not legal:
            raise IllegalActionError(
                f"The engine offered seat {self.player_id} no legal actions"
            )

        self.decisions += 1
        action = self._decide(env, legal)

        # Fail closed. The engine is the authority on legality; if the policy
        # disagrees with it, that is a bug to fix, not a move to paper over.
        if action not in legal:
            raise IllegalActionError(
                f"Seat {self.player_id} chose {action} "
                f"({action_to_description(action)}), which is not legal here. "
                f"Engine offered {len(legal)} actions: {legal[:16]}"
                f"{'...' if len(legal) > 16 else ''}"
            )
        return action

    def describe(self) -> dict[str, Any]:
        """Everything needed to identify this agent in a results table."""
        return {
            "checkpoint": str(self.checkpoint),
            "sha256": checkpoint_sha256(self.checkpoint),
            "player_id": self.player_id,
            "hybrid": self.hybrid,
            "restrict_liquidation": self.restrict_liquidation,
            **{
                key: self.metadata[key]
                for key in ("hidden_dim", "games_trained", "step_count", "ruleset")
            },
        }

    def __repr__(self) -> str:
        return (
            f"SubmissionAgent(player_id={self.player_id}, "
            f"checkpoint={self.checkpoint.name!r}, hybrid={self.hybrid}, "
            f"restrict_liquidation={self.restrict_liquidation})"
        )

    # ── Decision path ─────────────────────────────────────────────────────────

    def _decide(self, env, legal: list[int]) -> int:
        pid = self.player_id
        candidates: Sequence[int] = legal

        if self.hybrid:
            # Rules 1 and 2. BUY_PROPERTY and ACCEPT_TRADE were never routed
            # through the actor during training, so they are not routed through
            # it now; the actor has no useful opinion about them.
            if _BUY in candidates and fixed_buy_decision(env, pid):
                self.hybrid_decisions += 1
                return _BUY
            if _ACCEPT in candidates and env._incoming_trade(pid) is not None:
                self.hybrid_decisions += 1
                return (
                    _ACCEPT
                    if fixed_accept_trade_decision(env, pid)
                    else _DECLINE  # offered alongside ACCEPT_TRADE by the engine
                )
            candidates = [a for a in candidates if a not in _HYBRID_FIXED]

        if self.restrict_liquidation:
            # Rule 3. Debt-forced liquidation is never blocked: when the engine
            # sets debt_player it offers liquidation as the only way to settle,
            # and restrict_actions leaves that case alone.
            narrowed = restrict_actions(env, pid, candidates)
            if len(narrowed) != len(candidates):
                self.restricted_decisions += 1
            candidates = narrowed

        if not candidates:
            # Only reachable if the hybrid filter emptied the list, which the
            # engine's own action menus make impossible (BUY_PROPERTY always
            # comes with END_TURN, ACCEPT_TRADE always with DECLINE_TRADE).
            # Falling back to the engine's list keeps the result legal either way.
            candidates = legal

        return self._argmax(env, pid, candidates)

    def _argmax(self, env, pid: int, candidates: Sequence[int]) -> int:
        state = np.asarray(env._get_state(pid), dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(
                f"Engine produced an unusable observation for seat {pid}: "
                f"shape={state.shape}, finite={bool(np.isfinite(state).all())}"
            )

        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[0, list(candidates)] = True
        with torch.inference_mode():
            log_probs = self.actor(torch.from_numpy(state).unsqueeze(0), mask)
        return int(log_probs.argmax(dim=-1).item())


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DEFAULT_CHECKPOINT",
    "CheckpointError",
    "IllegalActionError",
    "SubmissionAgent",
    "checkpoint_sha256",
]
