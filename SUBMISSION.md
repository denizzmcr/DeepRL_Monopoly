# SUBMISSION.md — what we are submitting and what it scores

Read `HANDOFF.md` for project state and `CLAUDE.md` for settled decisions. This
file is narrower: it describes the artifact that gets submitted, identifies it
exactly, and records the conditions under which every number below was measured.

---

## 1. The entry point

```python
from submission_agent import SubmissionAgent

agent = SubmissionAgent(player_id=seat)      # defaults to artifacts/CHAMPION.pt
action = agent.choose_action(env)            # -> one legal action index
```

`submission_agent.py` sits at the repository root and is the only file the match
harness needs. It runs on CPU under `eval()` + `torch.inference_mode()`, loads
only the actor (the critic and optimizer state in the checkpoint are training
machinery), and is deterministic: it takes the argmax over legal actions rather
than sampling, with ties breaking to the lowest action index.

`player_id` is the seat being played and need not match the seat the checkpoint
trained in. Training rotated the learner through all four seats and the
observation is built per player, so the weights are seat-agnostic.

**It never substitutes.** If the action it picks is not in
`env.get_allowed_actions(player_id)`, it raises `IllegalActionError` rather than
falling back to `END_TURN`. A silent fallback would make a broken policy look
like a working one while playing something nobody measured.

**It never imports `ASU_FROZEN_TEACHER`.** The inference path is written out
locally instead of reusing `ASU_FROZEN_TEACHER.evaluate._NeuralAdapter`, which is
otherwise the same algorithm. `tests/test_submission_agent.py` imports the module
in a fresh interpreter and fails if `ASU_FROZEN_TEACHER`, `monopoly_bench`,
`RL_CFR_MONOPOLYMODIFIED` or `SLM_HANDMADE_MONOPOLY` appears in that
interpreter's `sys.modules`; a second test asserts the two paths choose the same
action at every decision of a full game, so the measured results below carry over
to the packaged agent.

---

## 2. Checkpoint identity

| field | value |
|---|---|
| path | `artifacts/CHAMPION.pt` (force-added to git; `artifacts/` is otherwise ignored) |
| **SHA-256** | `a3ad0837a76ef2f41bd2b48ba5fc94c8b1c261cf92bff77865e844154b17f228` |
| size | 14,204,135 bytes |
| ruleset | `ppo-plus-v2` |
| checkpoint format | 3 |
| state dim / action dim | 300 / 2958 |
| network | `ActorNetwork`, 3 × (Linear 256 → LayerNorm → ReLU) → 2958 |
| hybrid | `True` |
| games trained | 4,000 (from scratch) |
| policy steps | 3,130,071 |
| trained in seat | 0 (with seat rotation) |

`training_config` as stored in the file:

```
gamma 0.99   lam 0.95   clip_eps 0.2   entropy_coef 0.005   value_coef 0.5
max_grad_norm 0.5   n_steps 1024   n_epochs 4   batch_size 64
win_loss_bonus 1.0   restrict_liquidation True
```

Verify with:

```powershell
venv\Scripts\python.exe -c "import submission_agent, json; print(json.dumps(submission_agent.SubmissionAgent(0).describe(), indent=1))"
```

### Provenance — which round-robin row this file is

`artifacts/diag/roundrobin2/summary.json` contains a candidate literally named
`champion`, and **it is not this file**. `artifacts/CHAMPION.pt` is the candidate
recorded there as **`w1_fresh_s66`**; the row named `champion` is the earlier
checkpoint it replaced. The two are easy to confuse and score very differently
against the trained trio (57.5% vs 83.75%).

This was settled by re-measurement rather than by reading names — see §5, where
three fields reproduce `w1_fresh_s66`'s numbers exactly and none reproduce
`champion`'s.

---

## 3. The three hardcoded rules

The rules allow at most five. We use three, all of them in `submission_agent.py`.

| # | rule | where |
|---|---|---|
| 1 | `fixed_buy_decision` — buy if it completes a monopoly, else if $100 would remain | `monopoly_game_engine/agent_ppo.py` |
| 2 | `fixed_accept_trade_decision` — accept if it completes a monopoly, else if net worth change ≥ 0 | `monopoly_game_engine/agent_ppo.py` |
| 3 | `restrict_actions` — never *voluntarily* mortgage or sell | `monopoly_game_engine/action_filters.py` |

Rules 1 and 2 are the hybrid split (`BUY_PROPERTY`, `ACCEPT_TRADE`) the network
was trained under. Those two decisions were never routed through the actor
during training, so they are not routed through it at match time either.

**Rule 3 decides the game.** Voluntary liquidation was masked out during
training, so those logits never received a gradient and are still at
initialisation. The same checkpoint run *without* the restriction scores **0%**
and liquidates 251 times per game. This is why the flag lives inside the
checkpoint (`training_config.restrict_liquidation`) and is read from there rather
than passed in: a checkpoint trained with the restriction is wrong to run
without it. Checkpoints predating the flag load as unrestricted, which is how
they were trained.

Debt-forced liquidation is never blocked. When the engine sets `debt_player` it
offers liquidation as the only way to settle, and `restrict_actions` leaves that
case alone.

---

## 4. Opponent identity

`fixed-a` … `fixed-f` are `monopoly_game_engine.agents_fixed.FP_AGENT_CLASSES`,
in order:

| id | class | buying behaviour |
|---|---|---|
| `fixed-a` | `TheHoarder` | only to complete a monopoly, or railroads, keeping $600 in reserve |
| `fixed-b` | `TheDealMaker` | buys if affordable with a $100 buffer; builds only above $800 surplus |
| `fixed-c` | `TheGambler` | buys every unowned property down to $50 cash |
| `fixed-d` | `TheBuilder` | only green, dark blue and railroads; saves capital to develop |
| `fixed-e` | `TheBlocker` | always buys to deny an opponent a monopoly, else normal |
| `fixed-f` | `TheRailBaron` | railroads and utilities only |

Field names used in the results tables:

| field | seats |
|---|---|
| trained trio | `fixed-a`, `fixed-b`, `fixed-c` |
| unseen trio | `fixed-d`, `fixed-e`, `fixed-f` |
| builders ×3 | `fixed-d` ×3 |
| dealmakers ×3 | `fixed-b` ×3 |
| blockers ×3 | `fixed-e` ×3 |
| vs a rival neural agent | the previous champion checkpoint, three seats |
| vs another rival neural agent | the `nohybrid_mixed` checkpoint, three seats |

The two rival-neural fields cannot be reproduced on Machine 2: those checkpoints
live under `artifacts/`, which is gitignored except for `CHAMPION.pt`. The exact
seat composition behind the labels "strong mix" and "blocker mix" in
`roundrobin2/summary.json` is **not recorded anywhere in the repository** — the
script that produced that file was never committed, only its output. Everything
else in the table above is confirmed.

"blockers ×3" is a Machine 2 field, not a rename of Machine 1's "blocker mix".
Three Blockers score 52.5% against us (§5.2) where "blocker mix" scores 57.5%,
on the same seeds and the same seat balancing — so the two are genuinely
different opponent sets, and the label really does mean a mix.

---

## 5. Measured results

### 5.1 As reported by Machine 1 (`HANDOFF.md`)

Seat-balanced, 80 games per cell, source `artifacts/diag/roundrobin2/summary.json`,
row `w1_fresh_s66`.

| field | win rate |
|---|---|
| trained trio (Fixed-A/B/C) | 57.5% |
| unseen trio (Fixed-D/E/F) | 68.8% |
| builders ×3 | 97.5% |
| dealmakers ×3 | 35.5% (2,500 games; the 31.2% cell below is 80 games) |
| strong mix | 67.5% |
| blocker mix | 57.5% |
| vs a rival neural agent | 35.0% |
| vs another rival neural agent | 28.7% |
| **worst case** | **28.7%** |

Parity in a four-player game is 25%. In a mixed field the agent beats the
strongest scripted agent 52% to 26%.

### 5.2 Independently re-measured on Machine 2

Same evaluator, same seeds, same seat balancing, run against
`artifacts/CHAMPION.pt` as committed. Outputs in `artifacts/diag/submission/`.

| field | opponents | wins / games | win rate | Wilson 95% | round cap hit | median rounds |
|---|---|---|---|---|---|---|
| trained trio | `fixed-a fixed-b fixed-c` | 46 / 80 | **57.50%** | 46.6 – 67.7 | 12.5% | 77 |
| unseen trio | `fixed-d fixed-e fixed-f` | 55 / 80 | **68.75%** | 57.9 – 77.8 | 5.0% | 58 |
| builders ×3 | `fixed-d fixed-d fixed-d` | 78 / 80 | **97.50%** | 91.3 – 99.3 | 0.0% | 30 |
| blockers ×3 | `fixed-e fixed-e fixed-e` | 42 / 80 | **52.50%** | 41.7 – 63.1 | 73.8% | 200 |
| dealmakers ×3 | `fixed-b fixed-b fixed-b` | 25 / 80 | **31.25%** | 22.2 – 42.1 | 67.5% | 200 |

Four of these fields have a counterpart in `roundrobin2/summary.json`, and all
four reproduce `w1_fresh_s66` **exactly** — 57.50, 68.75, 97.50, 31.25 — while
none matches the row named `champion` (83.75, 58.75, 97.50, 31.25). The trained
trio alone separates them by 26 points. That is what identifies the file.
"blockers ×3" is new here and has no counterpart to compare against.

### 5.3 Seed sets

`ASU_FROZEN_TEACHER.evaluate` takes one seed per **paired four-game block**: for
each seed it plays the game four times with the focus agent in each of the four
seats, so no result depends on turn order.

* **Seeds used: 0 – 19**, i.e. 20 blocks × 4 seats = **80 games per field**.
* The same seed set was used for every field, so the fields are paired with each
  other as well.
* The engine is seeded before `MonopolyEnv` construction, so the shuffled turn
  order is part of the seed.
* The training seed is not stored in the checkpoint; the run name `w1_fresh_s66`
  implies 66, but that is an inference from a filename, not a recorded fact.

### 5.4 Truncation

Two different caps exist and they are easy to conflate.

* **Round cap** — `max_rounds=200`. The game ends and the winner is decided on
  net worth. `env.done` is set, so this is a *completed* game as far as the
  evaluator is concerned.
* **Decision cap** — `max_decisions=20000` in the evaluator. A game that hits
  this is discarded from win-rate denominators. The field reported as
  `truncations` in the evaluator's JSON is this one, **not** the round cap.

Measured on Machine 2 across all five fields: the decision cap was hit **0 times
in 400 games**, and the busiest single game used 9,965 decisions of the 20,000
available — so no result below is distorted by a discarded game. The round cap
was reached in 127 of those 400 games (31.8%), concentrated almost entirely in
the two stalling fields. Per-field rates are in §5.2.

Machine 1's figures for the round cap, from 2,500-game exports: **73%** against
three Deal-Makers (mean 172 rounds) versus **~15%** against Fixed-A/B/C. Our
12.5% on 80 games against Fixed-A/B/C is consistent with that.

Against three Deal-Makers, hitting the cap is the normal outcome rather than a
failure: three aggressive traders keep the board fragmented, nobody assembles a
dominant position, 0.92 of 4 players are bankrupt at the end, and the game is
decided on net worth among four survivors. 35.5% of a near-coin-flip among four
is above the 25% share.

Three Blockers do the same thing, which had not been measured before: **73.8%**
of those games run to the cap, median exactly 200 rounds, and we still take
52.5%. An opponent field that denies monopolies stalls the game rather than
beating us, and stalled games are decided on net worth, where we are ahead.

### 5.5 Per-move latency

Measured by `tests/test_submission_agent.py` over **2,725 moves** — three full
games, seats 0/1/2, against Fixed-A/B/C, on the Windows machine (8 cores, CPU
torch 2.13, no GPU), otherwise idle:

| p50 | p95 | max |
|---|---|---|
| 1.02 ms | 2.04 ms | 50.4 ms |

A second run on the same machine gave p50 0.88 ms / p95 2.13 ms / max 72.1 ms, so
p95 is stable to about 0.1 ms. The maximum is the first call, which pays one-off
kernel setup. The per-move budget in the competition is unknown; the design
target is ~1 s, and the test asserts p95 stays under it as a regression guard —
the measured margin is roughly 500×.

---

## 6. Reproducing the numbers

```powershell
# identity
venv\Scripts\python.exe -c "import submission_agent as s; print(s.checkpoint_sha256(s.DEFAULT_CHECKPOINT))"

# the packaging tests, including ASU isolation and the p95 latency report
venv\Scripts\python.exe -m pytest tests\test_submission_agent.py -q -s

# one evaluation cell (80 games, seat-balanced)
venv\Scripts\python.exe -m ASU_FROZEN_TEACHER.evaluate `
  --focus "ppo:artifacts/CHAMPION.pt" --opponents fixed-a fixed-b fixed-c `
  --seeds (0..19) --output artifacts\diag\submission\trained_trio.json
```

`ASU_FROZEN_TEACHER.evaluate` is an evaluation harness, not part of the
submission. Using it as a benchmark is allowed; the submitted agent never
imports it.

Test suite status on Windows: **110 passed, 2 failed**. Both failures are in
`tests/test_gemma4_notebook.py`, both predate this work, and both are platform
artifacts — they assert POSIX permission bits (`st_mode & 0o077`) that Windows
cannot represent. The same suite is 97 passed / 0 failed on macOS. Excluding
those two, the Windows baseline was 95 passed before this work and 110 after,
the difference being 8 submission tests and 7 parallel-collection tests.

---

## 7. Known limits

* **Non-transitivity is large and real.** A previous champion scored 36.5% and
  19.0% in two four-player fields that differed only in the fourth player. No
  single win rate here predicts the match, and the other three teams' agents
  cannot be seen in advance. The worst case in §5.1 (28.7%, against a rival
  neural agent) is the number to plan against, not the 97.5%.
* The two rival-neural fields and the "strong mix" / "blocker mix" compositions
  are not reproducible from this repository (§4).
* The agent is deterministic. An opponent that could observe it across many games
  could in principle exploit that; within a round-robin of unseen agents this is
  not a practical risk, and determinism buys reproducibility.
