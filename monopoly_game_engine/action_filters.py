"""Policy-side action restrictions.

These filter the candidate actions a *policy* will consider. They never touch
the engine's legality rules: `env.get_allowed_actions` stays the authority on
what is legal, and these only narrow what the agent chooses to look at.

Why this exists
---------------
Measured on 2026-08-10, 200 games, agent in seat 0 against Fixed-A/B/C:

    random over all legal actions                        0.0% wins
    random, no trade offers                              0.0% wins
    random, no trade offers and no voluntary liquidation 21.0% wins

The action space contains moves that are almost always self-destructive:
mortgaging, selling houses and hotels, and selling deeds back to the bank. A
stochastic policy keeps sampling them and dismantles its own position, which is
why a 2,000-game PPO run scored 0-2.5% while a random policy that simply
refuses to liquidate scores 21%.

Liquidation is only ever blocked when it is *voluntary*. When the engine sets
`debt_player`, it returns liquidation actions exclusively as the way to settle
the debt, and those are never filtered.
"""

from .actions import OFFSETS

# Mortgaging, and selling houses, hotels, or deeds back to the bank. Deliberately
# excludes unmortgage and improve_house/improve_hotel, which build value.
_LIQUIDATION_RANGES = (
    (OFFSETS["mortgage"], OFFSETS["unmortgage"]),
    (OFFSETS["sell_house"], OFFSETS["buy_trade"]),
)
_TRADE_OFFER_RANGE = (OFFSETS["buy_trade"], OFFSETS["auction"])


def _in_ranges(action: int, ranges) -> bool:
    return any(low <= action < high for low, high in ranges)


def is_liquidation(action: int) -> bool:
    """True for mortgage, sell house/hotel, and sell-deed-to-bank actions."""
    return _in_ranges(action, _LIQUIDATION_RANGES)


def is_trade_offer(action: int) -> bool:
    """True for the cash-for-property and property-for-property offer actions."""
    return _in_ranges(action, (_TRADE_OFFER_RANGE,))


def restrict_actions(
    env,
    player_id: int,
    allowed,
    *,
    block_voluntary_liquidation: bool = True,
    block_trade_offers: bool = False,
):
    """Narrow ``allowed`` to the actions a policy should consider.

    Returns the original list unchanged if filtering would leave nothing, so a
    restriction can never produce an empty choice set or an illegal fallback.
    """
    forced_liquidation = getattr(env, "debt_player", None) == player_id
    drop_liquidation = block_voluntary_liquidation and not forced_liquidation

    kept = [
        action
        for action in allowed
        if not (drop_liquidation and is_liquidation(action))
        and not (block_trade_offers and is_trade_offer(action))
    ]
    return kept or list(allowed)
