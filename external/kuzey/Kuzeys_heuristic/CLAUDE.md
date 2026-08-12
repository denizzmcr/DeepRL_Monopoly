# Kuzey's Monopoly heuristic

A hand-written agent for the simplified board Monopoly used in this competition. It is the
strongest heuristic produced so far, and it is self-contained: this folder is everything you
need to run it.

```python
from heuristic import Champion

agent = Champion()
action = agent.choose_action(game, player_id, decision_seed)
```

That is the whole interface. `game` may be the raw `MonopolyEnv` or any wrapper with an
`.env` attribute. It returns a legal action id. No training, no checkpoint, no GPU, no torch.

---

## Check it works first

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python play.py
```

Must print **`champion vs mixed:  79/120 = 65.8%`** and `SELF-TEST: ... PASS`. That is the
number this agent was measured at, reproduced from scratch. If it prints anything else, do
not trust anything below — the package is not the agent that was measured.

Runs in about two minutes on one core. Needs Python 3.10+ and **nothing else** — no numpy, no
torch, no install step.

---

## What it scores

Win rate as one player against three opponents. **25% is even** — there are four seats.

| opponents | this agent | previous best | measured on |
|---|---|---|---|
| `asu_class` ×3 | **63.6%** | 58.9% | 360 games |
| `asu_class`, builder, blocker | **75.0%** | 60.0% | 360 games |
| builder, gambler, blocker | **68.3%** | 52.2% | 360 games |
| 3 copies of the previous champion | **45.0%** | 33.3% | 360 games |

Better on every panel, no panel where it is worse. Those 360-game numbers are from a run on
dice it was **not** selected on, so they are not the numbers it was tuned to.

A fifth panel it had never seen (dealmaker, builder, blocker) gave +12.5 points over the
previous champion, so the gain is not specific to one set of opponents.

> The `play.py` self-test uses `builder, gambler, blocker` at 120 games and gives 65.8%,
> slightly under the 68.3% above, because that row is a larger run on different dice. Both
> are real; the 360-game one is the better estimate.

---

## What it actually does

Three changes on top of a plain rule-based agent. All three are about the same thing: **the
agent was sitting on money it should have been spending.**

**1. Spend down to zero when building.** It used to hold half its cash reserve back before
buying a house. Now it doesn't. Worth about +5 points on its own, and it replicated on a
second independent set of dice.

**2. The war chest — this is the main one.** Some properties can *never* become a complete
colour set, because an opponent already owns one of the group and the engine only grants a
monopoly to someone holding all of them. Those properties are dead money. So: mortgage them,
and hold the cash to bid on properties that are still available. They stay mortgaged while
the board is still being bought, otherwise the agent redeems next turn what it mortgaged last
turn and just oscillates.

The one number in it is not tuned: **654 auctions were measured clearing at a median of
exactly 1.00× list price**, so list price is what it takes to be in the auction at all.

Why it works: **the board is bought out by round 25 and frozen by round 50.** There is a
narrow window where cash converts into property at all, and a permanently dead deed is worth
more as bidding power inside that window than as an asset that can never become a set.

**3. Near the round cap, buy properties for cash at list price.** Included but **not part of
the claim.** At the 200-round cap the winner is whoever has the highest net worth, and cash
counts 1.0× while an unmortgaged property counts 2.5× (5.0× in a complete set). So converting
cash into property before the cap is close to free.

It only works against opponents who will accept a cash-for-property trade, and **six of the
seven opponents refuse by construction** — only `TheGambler` accepts, because its accept test
compares list prices and a purchase at list scores exactly zero. On every panel where it
cannot fire it is *outcome-identical*, so it costs nothing. `Champion` excludes it;
`ChampionPlus` includes it.

---

## What you get

```python
from heuristic import Champion, ChampionPlus, SpineH100, Spine
```

| | what it is | use it for |
|---|---|---|
| `Champion` | changes 1 + 2 | **the agent.** This is the one. |
| `ChampionPlus` | + change 3 | a free extra that only pays against trading opponents |
| `SpineH100` | the previous champion | a strong-but-beatable training opponent |
| `Spine` | the base, untuned | a weak rung — it buys almost nothing |

**As a training ladder**, the useful rungs go roughly: `Spine` → `SpineH100` → `Champion`.
Ratings from a 1,600-game round robin over 21 agents (Elo-like, `SpineH100` pinned at 0):
`Champion` **+131**, `SpineH100` **0**, and for reference the ASU-class agent sits at **+1** —
level with `SpineH100` in a mixed field.

---

## Layout

```
Kuzeys_heuristic/
  CLAUDE.md            this file
  play.py              runs games and prints the win rate; also the self-test
  heuristic/
    __init__.py        exports Champion, ChampionPlus, SpineH100, Spine
    spine.py           the base agent (1,191 lines) — board maths and all decisions
    champion.py        the three changes, as mixins over the base
    _bind.py           makes `import monopoly_game_engine` resolve to engine/
  engine/              the game engine, pinned copy, never edited
```

---

## Three things that will bite you

**`PYTHONHASHSEED=0` or the engine is not reproducible.** Four identical runs gave win totals
[15, 15, 16, 15] with different per-game results. Something in the decision path iterates a
container keyed by strings, and Python randomises string hashing per process. It flips about
one game in forty — a ~2.5 point drift, which is exactly the size that invents a fake
improvement. `play.py` re-execs itself with it pinned so you can't forget; if you write your
own harness, set it.

**Rotate the seats.** Turn order is worth real win rate. Measuring in one seat measures the
seat as much as the agent. `play.py` plays each block of four games with the agent in each
seat once, on the same dice.

**The shipped opponents emit illegal actions.** They fall through to END_TURN in the
raise-funds state where END_TURN isn't legal, and `env.step` then raises. Unsanitised, 8 of 8
games are destroyed. Clamp them (prefer END_TURN, else the first legal action). **This agent
never needs clamping** — `play.py` counts it and treats any clamp as a hard failure.

---

## What it is not

It is **not a submission**. The competition's 95% rule means the entry has to be a model, and
this is hand-written rules. Its role is to be the *teacher* a network learns from, and to be a
strong sparring opponent — not the thing that gets handed in.

It is also **not tested against a real rival**. Every number here is against fixed archetypes
and copies of itself. The actual opponents are four other teams' agents, and nothing here
predicts how it does against those.

## Known weakness, if you want to improve it

At round 25, the games it eventually wins and the games it eventually loses have **the same
number of properties** — 9.0 either way. The whole difference is how many are in a *completed*
set: **3.0 in wins, 1.0 in losses.** It is not short of property. It is short of *finished
colour groups*, and closing them sooner is the clearest remaining gain.
