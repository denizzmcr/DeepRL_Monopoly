"""Candidate rules aimed at the weakness the competitor field exposed.

The diagnosis (``docs/GAUNTLET.md``, 240 instrumented games): we do not lose by
dying more than the field -- our bankruptcy rate is close to 6c0de's, and they
win twice as often. We lose because when we survive we hold a third of the
leaders' net worth, having built 6.4 houses a game against their 24.1, off
14.9 deeds against their 21.5.

Fewer deeds is upstream of fewer houses, and the auction ceiling is the most
likely reason we hold fewer deeds. The shipped constants are

    AUCTION_MAX_MULT   = 2.50   # never bid above this multiple of list
    AUCTION_PLAIN_MULT = 0.85   # a deed we merely like is only worth a bargain

and the second is the suspect. Declining to buy at list in order to win the
auction cheaply is a strategy that pays only when the other bidders are weak;
in this field it means somebody else takes the deed. The engine's own scoring
rule says what a deed is actually worth:

    unmortgaged deed = 2.5x list      inside a completed group = 5.0x list
    cash             = 1.0x

so paying X for a plain deed changes net worth by 2.5*price - X: break-even is
at 2.5x list, exactly ``AUCTION_MAX_MULT``. A cap of 0.85x is far under the
break-even the engine defines, and it is the *plain* cap that binds most often.

Cash is not worthless -- it is what stops you going bankrupt -- so the right
ceiling is below the 2.5x break-even, not at it. These variants move it, and
the field decides. Nothing here is tuned to a particular opponent: every
constant is anchored to the engine's scoring rule, which is fixed and identical
for everyone, so an improvement should survive the other teams changing their
agents.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_H = str(REPO / "underdog")
if _H not in sys.path:
    sys.path.insert(0, _H)

from heuristic.champion import StChestScrap  # noqa: E402


class PlainUp(StChestScrap):
    """Raise only the plain-deed auction ceiling toward the engine's break-even."""
    name = "v2_plain"
    AUCTION_PLAIN_MULT = 1.30


class PlainUp2(StChestScrap):
    name = "v2_plain2"
    AUCTION_PLAIN_MULT = 1.75


class BuyMore(StChestScrap):
    """Spend more freely at list price.

    ``_should_buy`` refuses whenever cash would drop under the reserve, and
    only relaxes to half the reserve for a deed worth >=1.5x its price. At
    2.5x score per dollar, a deed at list is the best rate on the board.
    """
    name = "v2_buy"
    BUY_RESERVE_RELIEF = 0.25


class PlainBuy(PlainUp, BuyMore):
    name = "v2_plainbuy"


class Horizon(StChestScrap):
    """Value deeds over a longer rent horizon, which raises every deed's value."""
    name = "v2_horizon"
    VALUE_HORIZON = 150.0


VARIANTS_V2 = {
    "v2_plain": PlainUp,
    "v2_plain2": PlainUp2,
    "v2_buy": BuyMore,
    "v2_plainbuy": PlainBuy,
    "v2_horizon": Horizon,
}


# --------------------------------------------------------------------------
# The measured cause, and the rule aimed at it
# --------------------------------------------------------------------------
# Instrumented over real games against this field: of every chance to buy an
# unowned deed, we took half and declined half -- and *every* decline was for
# low value, none for the cash floor. The median deed was valued at 0.91x its
# list price, so `_should_buy`'s final test, `value >= price`, rejected it.
#
# That test is measuring the wrong thing. `Ctx.deed_value` is rent income over
# a horizon, but `Property.calculate_net_worth` -- the function that decides a
# capped game and underwrites every liquidation -- prices an unmortgaged deed
# at 2.5x list, or 5.0x inside a completed group, against cash at 1.0x. Buying
# at list therefore converts $P of score into $2.5P of score whatever the rent
# does, and declining is a guaranteed loss of 1.5x the price in the only
# currency the engine ranks players by.
#
# `_score_value` already implements exactly that rule, and the shipped agent
# already consults it -- but only through `_endgame_weight`, which is zero
# until round 100. By then the board is bought. These variants let a deed be
# worth what the engine says it is worth for the whole game.
#
# The reserve test is deliberately left in place. Cash is what stops you going
# bankrupt, and we already go bankrupt in 68% of games, so this raises the buy
# rate without removing the floor that survives it.

from heuristic.spine import _PRICE_OF, _AUCTION_PASS  # noqa: E402
from monopoly_game_engine.actions import AUCTION_ACTION_TO_INCREMENT  # noqa: E402


class _ScoreBuyMixin:
    """Value an unowned deed at what it scores, not only at the rent it earns."""

    def _score_of(self, ctx, square: int) -> float:
        return self._score_value(ctx, square, ctx.pid)

    def _should_buy(self, ctx, square: int) -> bool:
        env, pid = ctx.env, ctx.pid
        prop = env.properties.get(square)
        if prop is None or prop.owner is not None:
            return False
        price = _PRICE_OF[square]
        cash = env.players[pid].cash
        value = max(ctx.deed_value(square, pid), self._score_of(ctx, square))
        critical = value >= 1.5 * price
        floor = ctx.reserve * (self.BUY_RESERVE_RELIEF if critical else 1.0)
        if cash - price < floor:
            return False
        return value >= price


class ScoreBuy(_ScoreBuyMixin, StChestScrap):
    name = "v2_score"


class ScoreAuction(_ScoreBuyMixin, StChestScrap):
    """Score-aware buying, and an auction ceiling drawn from the same rule.

    The shipped ceiling is `min(value, 0.85 * price)` for a deed whose rent
    value is under list. With the deed scoring 2.5x, the honest ceiling is a
    fraction of that break-even rather than a fraction of list -- passing at
    0.85x hands a 2.5x asset to whoever bids 0.9x.
    """
    name = "v2_score_auc"
    AUCTION_SCORE_MULT = 1.40

    def _auction(self, ctx, allowed, aset: set) -> int:
        env, pid = ctx.env, ctx.pid
        square = env.auction_property_id
        if square is None or pid != env.auction_current_pid:
            return _AUCTION_PASS if _AUCTION_PASS in aset else allowed[0]

        price = _PRICE_OF[square]
        value = max(ctx.deed_value(square, pid), self._score_of(ctx, square))
        ceiling = min(value, self.AUCTION_SCORE_MULT * price)
        ceiling = min(ceiling, env.players[pid].cash - self.AUCTION_RESERVE)
        if ceiling <= 0:
            return _AUCTION_PASS

        options = []
        for action, increment in AUCTION_ACTION_TO_INCREMENT.items():
            code = int(action)
            if code not in aset:
                continue
            bid = env.auction_current_bid + increment
            if bid <= ceiling:
                options.append((increment, code))
        if not options:
            return _AUCTION_PASS if _AUCTION_PASS in aset else allowed[0]
        options.sort()
        return options[0][1] if self.AUCTION_MIN_RAISE else options[-1][1]


VARIANTS_V2["v2_score"] = ScoreBuy
VARIANTS_V2["v2_score_auc"] = ScoreAuction


class ScoreBuyDeep(_ScoreBuyMixin, StChestScrap):
    """Score pricing, and a thinner cash floor now that the value gate passes.

    ``BUY_RESERVE_RELIEF`` never bound before: the ``value >= price`` test
    rejected the deed first, so loosening the floor alone changed nothing
    (measured: 0.0 pp). With the value gate corrected the floor is what binds,
    so it is worth one measurement of its own.
    """
    name = "v2_score_deep"
    BUY_RESERVE_RELIEF = 0.25


VARIANTS_V2["v2_score_deep"] = ScoreBuyDeep
