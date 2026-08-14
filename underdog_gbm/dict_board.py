"""Harness JSON -> engine-shaped board.

The tournament harness hands every decision over a pipe as plain JSON:
``state`` is a dict, there is no live engine, and attribute access on it
returns nothing. The policy in this package reads a board through the engine's
own surface (``env.properties[sq].owner``, ``env.players[i].cash``,
``env.phase`` ...), so rather than rewrite 550 lines of feature code against a
second shape, this rebuilds that surface from the dict.

It does so with the engine's *real* ``Property`` and ``Player`` classes, not
stand-ins. Both are constructible from an id alone and carry only four and six
mutable fields respectively, all of which the harness sends. That means
``prop.get_rent(...)``, ``prop.is_real_estate`` and ``player.net_worth()``
behave in play exactly as they did in training, which a hand-rolled mock could
not promise.

Two fields the harness does not send are reconstructed rather than guessed:

``debt_amount``
    Not in the payload. The engine only withholds ``END_TURN`` during
    ``post_roll`` when a debt is outstanding, so presence of debt is exact;
    the amount is not recoverable and is left at zero. One of 71 features
    degrades, the debt *flag* that drives the liquidation branch does not.

``turn_order``
    Shuffled per game inside the engine and never transmitted. Only the
    observation builder needs it, and the observation is supplied directly as
    ``state["vector"]``, so nothing that runs here consumes it. Seat order is
    left canonical.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import engine_shim  # noqa: F401  (binds the engine package)
from engine.constants import PROPERTIES
from engine.state import Player, Property

__all__ = ["DictBoard", "board_from_state", "action_names", "in_debt"]

_NUM_PLAYERS = 4


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def action_names(state: Any) -> dict[int, str]:
    """``{action_id: readable_name}`` from ``state["actions"]``, ints keyed.

    The harness keys this by string because JSON object keys are strings.
    Everything downstream compares against ``allowed_actions``, which is ints.
    """
    if not isinstance(state, dict):
        return {}
    raw = state.get("actions")
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for key, name in raw.items():
        try:
            out[int(key)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def in_debt(state: Any, names: Optional[dict[int, str]] = None) -> bool:
    """True when the engine is withholding ``END_TURN`` to force liquidation.

    This is the exact condition, not a heuristic: the engine offers
    ``END_TURN`` in every ``post_roll`` that is not settling a debt.
    """
    if not isinstance(state, dict):
        return False
    board = state.get("board")
    phase = board.get("phase") if isinstance(board, dict) else None
    if phase != "post_roll":
        return False
    names = action_names(state) if names is None else names
    if not names:
        return False
    return not any(n.strip().upper() == "END_TURN" for n in names.values())


class DictBoard:
    """The subset of the engine surface the policy actually reads."""

    def __init__(self, board: dict, allowed: Sequence[int], seat: int,
                 debt: bool = False) -> None:
        self.round = _int(board.get("round"))
        self.phase = str(board.get("phase") or "post_roll")
        self.done = bool(board.get("done"))
        self.houses_available = _int(board.get("houses_available"), 32)
        self.hotels_available = _int(board.get("hotels_available"), 12)
        self.max_rounds = 200
        self.has_rolled = bool(board.get("has_rolled"))
        dice = board.get("last_dice")
        self.last_dice = (tuple(_int(d) for d in dice)
                          if isinstance(dice, (list, tuple)) and len(dice) == 2
                          else (1, 1))
        # Not transmitted; see module docstring.
        self.turn_order = list(range(_NUM_PLAYERS))
        self.debt_amount = 0.0

        self._seat = _int(seat)
        self._allowed = [int(a) for a in (allowed or [])]

        self.properties = self._build_properties(board)
        self.players = self._build_players(board)
        self._active = _int(board.get("active_player"), self._seat)
        self.current_turn_idx = self._active % _NUM_PLAYERS

        self._read_auction(board.get("auction"))

        # Never transmitted and not derivable. Defaults are the engine's own
        # "nothing pending" values, so every feature that reads them lands on
        # the same number it would have during an ordinary turn rather than on
        # a value the model never saw in training.
        self.consecutive_doubles = 0
        self.extra_roll_pending = False
        self.pending_trades: dict = {}
        self.debt_player = self._seat if debt else None
        self.debt_creditor = None

        if debt:
            # Amount is unrecoverable; a positive value is what marks the
            # liquidation branch as live for any feature that reads it.
            self.debt_amount = 1.0

    def _read_auction(self, auction: Any) -> None:
        """Mirror the engine's auction fields, present only during an auction."""
        self.auction = auction if isinstance(auction, dict) else None
        a = self.auction or {}
        prop_id = a.get("property_id")
        self.auction_property_id = None if prop_id is None else _int(prop_id)
        bidders = a.get("bidders")
        self.auction_bidders = ([_int(b) for b in bidders]
                                if isinstance(bidders, (list, tuple)) else [])
        cur = a.get("current_bidder")
        self.auction_current_pid = None if cur is None else _int(cur)
        self.auction_high_bid = _int(a.get("high_bid"))
        high = a.get("high_bidder")
        self.auction_high_bidder = None if high is None else _int(high)

    def active_player_id(self) -> int:
        """The seat whose turn it is, as the engine reports it."""
        return self._active

    def _incoming_trade(self, player_id: Any = None) -> None:
        """A method on the engine, so a method here.

        The harness sends no trade contents, so this is always "nothing
        pending". The consequence is bounded and known: the observation's
        trade block reads as empty, and the trade-decoding features fall to
        their no-offer values. It is the one place where the ported board
        carries less than the live engine did, and it is a limit of the
        payload, not of the port.
        """
        return None

    # ---------------- construction ----------------

    @staticmethod
    def _build_properties(board: dict) -> dict[int, Property]:
        """Every deed on the board, overlaid with whatever the payload sent.

        Built from ``PROPERTIES`` rather than from the payload alone so a
        short or partial ``board["properties"]`` cannot silently shrink the
        board and change group-completion arithmetic.
        """
        props: dict[int, Property] = {sq: Property(sq) for sq in PROPERTIES}
        sent = board.get("properties")
        if not isinstance(sent, (list, tuple)):
            return props
        for entry in sent:
            if not isinstance(entry, dict):
                continue
            sq = _int(entry.get("square_id"), -1)
            prop = props.get(sq)
            if prop is None:
                continue
            owner = entry.get("owner")
            prop.owner = None if owner is None else _int(owner)
            prop.mortgaged = bool(entry.get("mortgaged"))
            prop.houses = _int(entry.get("houses"))
            prop.is_monopoly = bool(entry.get("is_monopoly"))
        return props

    def _build_players(self, board: dict) -> list[Player]:
        players = [Player(i) for i in range(_NUM_PLAYERS)]
        sent = board.get("players")
        if isinstance(sent, (list, tuple)):
            for entry in sent:
                if not isinstance(entry, dict):
                    continue
                pid = _int(entry.get("player_id"), -1)
                if not 0 <= pid < _NUM_PLAYERS:
                    continue
                p = players[pid]
                p.cash = _float(entry.get("cash"))
                p.position = _int(entry.get("position"))
                p.in_jail = bool(entry.get("in_jail"))
                p.jail_turns = _int(entry.get("jail_turns"))
                p.gooj_card = bool(entry.get("gooj_card"))
                p.bankrupt = bool(entry.get("bankrupt"))
        # Ownership is taken from the deeds, not from each player's own list:
        # the deed table is the one the features read, so deriving both from it
        # keeps them from disagreeing.
        for p in players:
            p.properties = [pr for pr in self.properties.values()
                            if pr.owner == p.player_id]
        return players

    # ---------------- engine surface ----------------

    def get_allowed_actions(self, player_id: Any = None) -> list[int]:
        return list(self._allowed)

    def whose_turn(self) -> int:
        return self._seat

    def current_player(self) -> Player:
        return self.players[self._seat]

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"DictBoard(round={self.round}, phase={self.phase}, "
                f"seat={self._seat}, actions={len(self._allowed)})")


def board_from_state(state: Any, allowed: Sequence[int],
                     seat: int) -> Optional[DictBoard]:
    """Build a board from a harness ``state`` dict, or ``None`` if absent.

    ``None`` means the payload genuinely carried no board. It does not mean
    "attribute lookup failed", which is what the previous detector reported
    for every decision of every game.
    """
    if not isinstance(state, dict):
        return None
    board = state.get("board")
    if not isinstance(board, dict):
        return None
    if "players" not in board or "properties" not in board:
        return None
    built = DictBoard(board, allowed, seat, debt=in_debt(state))
    # The 300-float observation, straight from the harness. Model A consumes
    # it; rebuilding it here would be wrong (see gbm_policy) and slower.
    vector = state.get("vector")
    if isinstance(vector, (list, tuple)) and len(vector) == 300:
        try:
            built.supplied_vector = [float(v) for v in vector]
        except (TypeError, ValueError):
            built.supplied_vector = None
    else:
        built.supplied_vector = None
    return built
