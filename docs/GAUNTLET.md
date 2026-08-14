# Where we actually stand against the other teams

Six other submissions for this competition were published. This is our agent
measured against all of them, on one shared engine, seat-balanced.

Reproduce with [`tools/gauntlet.py`](../tools/gauntlet.py). Raw per-game records
are written to `artifacts/gauntlet_*.json`.

Parity in a four-player game is **25%**. All intervals are Wilson 95%.

---

## The result (2026-08-14, current submission)

Balanced round-robin against every rival's **then-current** code, pulled the
same day. Every one of the 35 four-agent subsets of the seven-agent field,
3 seeds each, all 4 seat rotations: 420 games, each agent appearing in 240 of
them against an identical distribution of opponents.

| rank | agent | team | win rate | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | **UNDERDOG (ours)** | this repository | **40.4%** | [34.4, 46.7] |
| 2 | 6c0de | `6c0de/exposure-monopoly-agent` | 35.0% | [29.2, 41.2] |
| 3 | inncenta | `Inncenta/monopoly` | 29.2% | [23.8, 35.2] |
| 4 | slayer | `emirkaanozdemr/monopoly` | 25.8% | [20.7, 31.7] |
| 5 | aline | `alinebidal10-afk/monopoly-competition-agent` | 23.8% | [18.8, 29.5] |
| 6 | expo | `emingurbuz9483/exposure-monopoly-algorithm` | 20.8% | [16.2, 26.4] |
| 7 | boom | `EnzeCbe/monopoly-boom` | 0.0% | [0.0, 1.6] |

Measured **through `agent.py` itself**, the file the harness loads, so seat
resolution, argument sorting and illegal-action substitution are all inside the
number. An earlier identical run that entered the policy directly scored 38.8%;
the two agree within noise, which is the point of re-running it.

**We finish first**, and the interval excludes parity. A larger run the same
day — 1,120 games over the 8-agent field including the previous submission —
reproduces it: ours 39.5% [35.5, 43.6], 6c0de 36.4%, and the previous
submission 21.8% at 6th of 8.

**What this result does not say.** Our lead over 6c0de is *not* statistically
separated: the intervals overlap almost entirely, and across the tables where
both sat down the split was 35.8% to 33.8%. The defensible claim is **level
with 6c0de, clear of the rest** — not that we are the strongest agent in the
field.

Integrity across both runs, all 1,540 games: **zero illegal actions, zero
exceptions, zero crashes, from any of the eight agents.**

### The field is moving

Three rivals pushed new code on 2026-08-14, and the changes were not cosmetic:

- **6c0de** replaced their previous entry — 103 lines wrapping `ASUValueV1` —
  with a 1,853-line self-contained policy titled *NEMESIS, ASU'suz* (ASU-free).
  The agent that finished first in the 2026-08-13 table below no longer exists.
  Their file records A/B results against tables labelled "us + aline + deniz +
  emir": they are benchmarking against us too.
- **expo** rewrote its decision policy (+187 lines): Markov landing odds,
  book-value acquisition, denial value, reserve fractions.
- **boom** merged ~20 commits of teacher distillation and a checkpoint
  refreshed on 32,647 games. It still wins 0 games.

Worth noting where the field converged: 6c0de and expo independently arrived at
the same principle our own `ChampionScore` fix used — price a deed at what
`Property.calculate_net_worth` *scores* it (2.5x list, 5.0x inside a group),
not at the rent it earns. Our submitted agent's feature set is built on the
same foundation.

**Any number here has a shelf life of about a day.** Re-run before relying on it.

---

## The previous result (2026-08-13), kept for the record

This is the measurement that motivated replacing the heuristic submission. Same
method, 600 games, the 15 four-agent subsets of a six-agent field, 10 seeds.

| rank | agent | team | win rate | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | 6c0de | `6c0de/exposure-monopoly-agent` | **37.0%** | [32.4, 41.8] |
| 2 | slayer | `emirkaanozdemr/monopoly` | 26.5% | [22.4, 31.0] |
| 3= | inncenta | `Inncenta/monopoly` | 25.0% | [21.0, 29.5] |
| 3= | aline | `alinebidal10-afk/monopoly-competition-agent` | 25.0% | [21.0, 29.5] |
| 5 | **UNDERDOG (heuristic, then submitted)** | this repository | **18.8%** | [15.2, 22.9] |
| 6 | expo | `emingurbuz9483/exposure-monopoly-algorithm` | 17.8% | [14.3, 21.8] |

**Fifth of six**, with the interval excluding 25%, so below parity was a real
effect and not sampling noise.

EnzeCbe's `monopoly-boom` was excluded from that table and scored separately: 0
of 400 games, and 0 of 40 against the engine's own fixed agents, so it is not
functional rather than merely weak, and leaving it in inflates everyone who
shared a table with it. The 2026-08-14 table above includes it because the
field is listed in full there; its 0.0% is the same finding.

## Head-to-head says something different, and is the less useful measurement

One of ours against three copies of one rival, 720 games each. The current
agent (2026-08-14) beside the heuristic it replaced (2026-08-13):

| opponent (3x) | current | 95% CI | previous |
| --- | ---: | --- | ---: |
| boom | 100.0% | [96.9, 100.0] | 100.0% |
| expo | 37.5% | [29.4, 46.4] | 35.8% |
| 6c0de | 34.2% | [26.3, 43.0] | 15.0% |
| inncenta | 29.2% | [21.8, 37.8] | 17.5% |
| slayer | 27.5% | [20.3, 36.1] | 26.7% |
| **aline** | **20.0%** | [13.8, 28.0] | **40.8%** |

Three of these rivals changed code between the two runs, so only `aline`,
`slayer` and `inncenta` are like-for-like. On those three the current agent is
much better against inncenta, level against slayer, and **much worse against
aline** — 20.0% where the heuristic took 40.8%.

That is worth stating plainly rather than burying: the new agent is not
strictly dominant, and the old one still beats it in one specific matchup.
Two things bound how much it matters. The 20.0% interval still contains parity,
so it is not a *losing* matchup, merely an unremarkable one. And facing three
clones is not the shape the competition uses — in the round-robin, where each
table holds four *different* agents, we lead aline 37.9% to 21.2%.

By the head-to-head table we beat every team. By the round-robin we are first
by a nose. Both are correct, and the round-robin is the one that matches the
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

## The gap was not a model-selection problem — it needed a different model

As of 2026-08-13, all three agents this project had produced, played through
*identical* tables and seeds, 200 games each:

| candidate | win rate | 95% CI |
| --- | ---: | --- |
| UNDERDOG — ChampionPlus, then submitted | 14.5% | [10.3, 20.0] |
| champion — base heuristic variant | 14.5% | [10.3, 20.0] |
| LAST_RESORT — learned checkpoint | 13.5% | [9.4, 18.9] |

Indistinguishable. Swapping among what existed at that point changed nothing,
which is what established that the distance to the top of the field could not
be closed by choosing differently among them — it needed a different model.
The gradient-boosted agent is that model, and it moved the same measurement
from 18.8% to 38.8%.

## What the top of the table is doing

As of 2026-08-13, two of the six called `ASU_FROZEN_TEACHER` inside their
submitted entry point:

- **6c0de** — `agent.py` was 103 lines wrapping `ASUValueV1`, plus a
  minimum-raise rule for auctions. It finished first.
- **inncenta** — its own candidate generator, scored by ASU's value function
  via `evaluate_value(env, pid)`.

The other four referenced ASU only in training or evaluation tooling, never on
the decision path.

**6c0de removed theirs on 2026-08-14**, replacing the ASU wrapper with a
self-contained policy whose module docstring names the change explicitly. So
the current field has one submitted entry point on the ASU decision path
(inncenta), not two.

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
# the competition shape: our agent + all six rivals, every 4-agent table
python tools/gauntlet.py --mode rr  --candidate LGBM --seeds 3  --workers 10

# two of ours in the same tables, so the comparison is paired
python tools/gauntlet.py --mode rr  --candidate LGBM --with UNDERDOG --seeds 4

python tools/gauntlet.py --mode h2h      --candidate LGBM --seeds 30 --workers 10
python tools/gauntlet.py --mode baseline --candidate LGBM --seeds 10 --workers 10
```

`LGBM` enters through `agent.py` itself — the same file the match harness
loads — rather than through `underdog_gbm.policy`, so what is measured includes
seat resolution, illegal-action substitution and the fallback path.

Pull the rivals' latest code first; three of them changed on the last day
measured:

```bash
for d in external/competitors/*/; do (cd "$d" && git pull --ff-only); done
```

The six repositories are cloned under `external/competitors/` (gitignored, not
redistributed here).
