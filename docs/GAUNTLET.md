# Where we actually stand against the other teams

Six other submissions for this competition were published. This is our agent
measured against all of them, on one shared engine, seat-balanced.

Reproduce with [`tools/gauntlet.py`](../tools/gauntlet.py). Raw per-game records
are written to `artifacts/gauntlet_*.json`.

Parity in a four-player game is **25%**. All intervals are Wilson 95%.

## The result

Balanced round-robin, 600 games. Every one of the 15 four-agent subsets of the
field, each played across 10 seeds and all 4 seat rotations, so every agent
plays exactly 400 games and meets the same distribution of opponents.

| rank | agent | team | win rate | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | 6c0de | `6c0de/exposure-monopoly-agent` | **37.0%** | [32.4, 41.8] |
| 2 | slayer | `emirkaanozdemr/monopoly` | 26.5% | [22.4, 31.0] |
| 3= | inncenta | `Inncenta/monopoly` | 25.0% | [21.0, 29.5] |
| 3= | aline | `alinebidal10-afk/monopoly-competition-agent` | 25.0% | [21.0, 29.5] |
| 5 | **UNDERDOG (ours)** | this repository | **18.8%** | [15.2, 22.9] |
| 6 | expo | `emingurbuz9483/exposure-monopoly-algorithm` | 17.8% | [14.3, 21.8] |

**We finish fifth of six.** The interval excludes 25%, so being below parity is
a real effect and not sampling noise. We are statistically tied with expo at the
bottom; the three agents above us are ahead by intervals that barely overlap
ours.

EnzeCbe's `monopoly-boom` is excluded from the table above and scored
separately: it wins 0 of 400 games, and 0 of 40 against the engine's own fixed
agents, so it is not functional rather than merely weak. Leaving it in would
inflate everyone who shared a table with it.

## Head-to-head says something different, and is the less useful measurement

One of ours against three copies of one rival, 720 games:

| opponent (3x) | our win rate | 95% CI |
| --- | ---: | --- |
| boom | 100.0% | [96.9, 100.0] |
| aline | 40.8% | [32.5, 49.8] |
| expo | 35.8% | [27.8, 44.7] |
| slayer | 26.7% | [19.6, 35.2] |
| inncenta | 17.5% | [11.7, 25.3] |
| 6c0de | 15.0% | [9.7, 22.5] |

By this table we beat three teams and tie a fourth. By the round-robin we are
fifth. Both are correct, and the round-robin is the one that matches the
competition format.

Facing three copies of a single agent is a different game from facing three
different ones: one systematic weakness in a mono-culture opposition is
exploitable three times over, and a policy tuned against a narrow field looks
much better than it is. In a mixed field our own weaknesses meet whichever
rival is best placed to exploit them. This is the non-transitivity recorded in
[`FINDINGS.md`](../FINDINGS.md), now measured against real opponents rather
than our own training league.

A third axis, against the engine's fixed agents — a reference nobody tuned
against — ranks us first at **85.0%**, ahead of expo (80.0%), aline and slayer
(65.0% each), and boom (0.0%). Beating a weak common reference does not predict
standing in a strong mixed field. Three axes, three different orderings.

## The gap is not a model-selection problem

All three agents this project produced, played through *identical* tables and
seeds, 200 games each:

| candidate | win rate | 95% CI |
| --- | ---: | --- |
| UNDERDOG — ChampionPlus, submitted | 14.5% | [10.3, 20.0] |
| champion — base heuristic variant | 14.5% | [10.3, 20.0] |
| LAST_RESORT — learned checkpoint | 13.5% | [9.4, 18.9] |

Indistinguishable. Swapping the submitted agent for either alternative changes
nothing, so the distance to the top of the field is not something a different
choice among what we have would close.

## What the top of the table is doing

Two of the six call `ASU_FROZEN_TEACHER` inside their submitted entry point:

- **6c0de** — `agent.py` is 103 lines wrapping `ASUValueV1`, plus a
  minimum-raise rule for auctions. It finishes first.
- **inncenta** — its own candidate generator, scored by ASU's value function
  via `evaluate_value(env, pid)`.

The other four reference ASU only in training or evaluation tooling, never on
the decision path.

We were instructed that ASU could be used as a training opponent and benchmark
but never implemented directly, so the entire project went into trying to
*beat* it rather than ship it. `FINDINGS.md` records where that ended:
distillation plateaued at 92% action agreement while still losing to ASU, and
PPO could not fine-tune past it. Finishing behind an agent that is ASU is
consistent with those findings.

This is a statement of what the code does, not a claim about anyone's
compliance. Different teams may have been given different constraints.

## Method, and why it is fair

**One engine for everybody.** Four of the six vendor their own
`monopoly_game_engine`. Python binds a module name once per process, so letting
each agent import its own copy means whichever loads first silently sets the
rules for the whole table. `_pin_engine()` imports our engine into
`sys.modules` before any competitor module is loaded, and the import cache then
takes precedence over `sys.path` — including over EnzeCbe's `agent.py`, which
inserts its own root at `sys.path[0]`. Verified: after loading all seven
agents, every `monopoly_game_engine.*` module resolves to this repository and
no competitor engine module is present.

Pinning is legitimate here because the parts that define the game are
byte-identical across all seven repositories — `state.py` and `actions.py` diff
to zero lines. The only engine edit anyone made is EnzeCbe's `_compute_reward`
liquidity-risk term, which is reward shaping read during training and never
during play.

**No team is scored on a harness mismatch.** There is no agreed interface, so
each adapter uses the call shape its author wrote: five take
`choose_action(env)`, EnzeCbe takes `choose_action(state, allowed_actions, env)`.
Two accommodations were needed:

- EnzeCbe's checkpoint stores `player_id: 0`, and `PPOAgent.load` rejects any
  checkpoint whose stored seat differs from the agent's — so their agent loads
  at seat 0 and nowhere else. The actor is seat-agnostic, so we load at seat 0
  and rebind the seat rather than score them zero for a metadata check.
- `ASU_SLAYER.policy` uses relative imports and must be imported as a package.

Illegal actions and raised exceptions are counted and substituted the way the
rest of this repository substitutes them, never treated as a forfeit. **Across
every run reported here, all seven agents produced zero illegal actions and
zero exceptions.**

**Seat balance.** Every seed is replayed with the lineup rotated through all
four seats, so seat advantage cancels exactly.

### A design error worth recording

The first melee build put UNDERDOG in every table (`UNDERDOG` + 3 rivals). That
asks a different question of each agent: ours faced three rivals, while every
rival faced ours plus only two. Those are different-strength fields and the
standings are not comparable — the bias ran against us, and it also inflated
every rival that shared a table with the non-functional entry. The `rr` mode
above replaced it: tables are drawn from all 4-subsets of the field, so no
agent is a fixed feature of the opposition. `artifacts/gauntlet_melee.json` is
kept for the record, but the `rr` numbers are the ones to cite.

## Commands

```bash
python tools/gauntlet.py --mode rr       --exclude boom --seeds 10 --workers 10
python tools/gauntlet.py --mode h2h      --seeds 30 --workers 10
python tools/gauntlet.py --mode baseline --seeds 10 --workers 10
python tools/gauntlet.py --mode melee    --exclude boom --candidate LAST_RESORT --seeds 5
```

The six repositories are cloned under `external/competitors/` (gitignored, not
redistributed here).
