# SUBMISSION.md — what we are submitting, and what it scores

This file describes the artifact that gets submitted, identifies it exactly, and
records the conditions under which every number below was measured. Read
[`docs/GAUNTLET.md`](docs/GAUNTLET.md) for the full standings against the other
teams and [`FINDINGS.md`](FINDINGS.md) for the project history behind them.

---

## 1. What is submitted

A repository URL and a pinned 40-character commit SHA. The harness clones that
commit and imports **`agent.py`** from the repository root. Nothing else is
uploaded, and no file needs to be named or configured anywhere.

```python
# what the harness calls
from agent import choose_action
action = choose_action(state, allowed_actions, env=env, player_id=seat)
```

`agent.py` also exports `Agent` and `make_agent(player_id)` for the class-form
calling convention. `env` and `player_id` are the optional extras the harness
passes by keyword; this agent declares them because its features read the board
directly and the 300-float state vector cannot be turned back into an
environment.

## 2. The policy

**Two gradient-boosted models over engineered features**, in `underdog_gbm/`.

| | |
|---|---|
| Model A | `underdog_gbm/models/model_a.txt`, 1,982,297 bytes — owns the **auction** family. Input: the 300-float observation plus a 36-dim action encoding. |
| Model B | `underdog_gbm/models/model_b.txt`, 2,500,790 bytes — owns **every other** family. Input: 71 engineered features per candidate action, trained with a LambdaRank objective over each decision's legal candidates. |

A dispatcher routes each decision by family; the agent takes the argmax over
legal candidates. Forced decisions (one legal action) skip the models entirely.

The 71 features are measurable quantities of the position — expected landings
per square from an exact Markov chain over the simulator's movement rules,
rent-income rates and dollar valuations under ownership / group-completion /
denial, group dynamics, liquidity and solvency, exact trade decoding with
both-sides valuations. No feature encodes a decision.

**Training.** Supervised on ~150,000 decision points (~11M candidate rows) from
large-scale self-play across varied opponent fields, against this project's own
strongest internal reference policy, with held-out games on disjoint seed blocks
for model selection. Held-out top-1 agreement 95.8% overall; the lowest family
is deed exchanges at 76.1%, measured to be dominated by near-equivalued
alternatives.

**This is a learned model**, not a hand-written rule set — the requirement in
`docs/CLAUDE.md` §1 — and it contains no ASU. See §6.

## 3. Measured strength

Against the six other teams' agents, all pulled the same day, on one shared
engine, seat-balanced. Parity is 25%, intervals are Wilson 95%. Full method in
[`docs/GAUNTLET.md`](docs/GAUNTLET.md).

**Competition shape** — every 4-agent subset of the 7-agent field, 3 seeds, all
4 seat rotations, 420 games:

| rank | agent | win rate | 95% CI |
| ---: | --- | ---: | --- |
| 1 | **ours** | **38.8%** | [32.8, 45.0] |
| 2 | 6c0de | 36.7% | [30.8, 42.9] |
| 3 | inncenta | 28.8% | [23.4, 34.8] |
| 4 | slayer | 25.8% | [20.7, 31.7] |
| 5 | aline | 23.8% | [18.8, 29.5] |
| 6 | expo | 21.2% | [16.5, 26.9] |
| 7 | boom | 0.0% | [0.0, 1.6] |

Reproduced on a larger 1,120-game run over an 8-agent field: ours 39.5%
[35.5, 43.6], 6c0de 36.4%, and the heuristic this replaced 21.8% (6th of 8).

**Honest limits on that number:**

- The lead over 6c0de is **not statistically separated** — the intervals overlap
  almost entirely and the direct split was 35.8% to 33.8%. "Level with 6c0de,
  clear of the rest" is what the data supports.
- Against **three copies of `aline`**, this agent takes 20.0% where the
  heuristic it replaced took 40.8%. It is not strictly dominant. The interval
  still contains parity, and the round-robin — four different agents per table,
  the shape the competition uses — has us ahead of aline 37.9% to 21.2%.
- **Three rivals changed code on the day this was measured.** Every number here
  has a shelf life of about a day.

## 4. Contract compliance

Verified by `tests/test_agent_entrypoint.py` (15 tests) and by the official
`python -m submission.validate`.

| requirement | how it is met |
|---|---|
| only legal actions | every return is checked against `allowed_actions` and replaced if absent. Measured: **0 illegal in 1,540 gauntlet games and 4 seat-rotated validator games.** |
| never touch the global RNG | the policy is a deterministic argmax and draws from no random source. A test snapshots `random.getstate()` and `np.random.get_state()` across a decision. |
| latency | p50 **1.3 ms**, p95 **6.6 ms**, max 7.6 ms over 594 decisions, single-core. |
| never raise | the entry point substitutes rather than propagating; in a scored match an exception and an illegal action lose equally. |
| no ASU at match time | §6. |

Note this is the one place the repository deliberately **does not** fail closed.
Everywhere else an illegal action raises, which is right during development
because it surfaces bugs. `agent.py` substitutes instead, because forfeiting a
match to prove a point is not a trade worth making.

## 5. Environment and packaging

The rules cap `requirements.txt` at **32 wheel-only PyPI entries**, resolve them
at validation time and ship the lock with the artifact; nothing is downloaded
during a match. Server agents run in Docker, Colab agents in a separate venv
capped at **2 GiB**.

```
numpy>=1.26,<3
lightgbm>=4.0,<5
```

Two entries of 32. Both publish wheels for every platform involved; neither
needs a compiler.

**torch is deliberately absent.** The submitted policy never calls it. The
engine imports torch on its own (`monopoly_game_engine/__init__.py:20`), but
that is the harness's dependency and the harness supplies it — listing it here
would pull the Linux CUDA wheel and exceed the 2 GiB sandbox by itself. A test
pins this reasoning and will fail if the engine ever stops importing torch.

**The one portability hazard, stated plainly.** LightGBM links OpenMP. Its
manylinux wheel vendors `libgomp`, so a Linux install — Docker and Colab, i.e.
both match environments — is self-contained. **macOS is different**: the wheel
does not vendor `libomp`, `pip install lightgbm` still succeeds, and then every
decision raises `OSError: Library not loaded: @rpath/libomp.dylib`. That is a
forfeit, not a slow game. On a Mac, `brew install libomp` first. This bit us
during development and is why §7 exists.

Checkout size: the working tree is ~7.6 MB against a 100 MB cap, of which 4.3 MB
is the two boosters. `directory_size()` excludes `.git`, so repository history
does not count.

## 6. The ASU prohibition

We were instructed that ASU may be used as a **training opponent and evaluation
benchmark** but never implemented directly, imitated, or run at match time.

- `grep -ri asu underdog_gbm/` returns nothing but the word "measured".
- `tests/test_agent_entrypoint.py` imports `agent` in a **fresh interpreter**,
  loads the policy, and asserts that no module under `ASU_FROZEN_TEACHER`,
  `monopoly_bench`, `RL_CFR_MONOPOLYMODIFIED` or `SLM_HANDMADE_MONOPOLY` is in
  that interpreter's `sys.modules` afterwards. A grep is not enough; a test
  fails loudly during a rushed merge.
- A companion test asserts the same fresh interpreter loaded the **real**
  policy, because an isolation guarantee that only holds for the fallback path
  guarantees nothing.
- The models were supervised on this project's own reference policy, not on ASU
  decisions. Distilling our own heuristic was explicitly approved; distilling
  ASU was not, and is not what happened.

ASU appears in `tools/gauntlet.py` only through the engine pin, and in the
evaluation harness. Using it as a benchmark is allowed; the submitted agent
never imports it.

## 7. The fallback, and why it exists

If LightGBM or either booster fails to load, `agent.py` falls back to
`ChampionScore` — the hand-written rule agent in `underdog/`, which needs no
dependency beyond the engine — and prints one line to `stderr`.

This is a real downgrade: `ChampionScore` measured 21.8% where the boosters
measure 38.8%. It is announced rather than absorbed silently, because a
submission that quietly plays a weaker policy than the one that was measured is
worse than one that says so. It is still preferred to the alternative, which is
losing every game to an import error.

`tests/test_agent_entrypoint.py` asserts the fallback is **not** in use, so this
path cannot go unnoticed in the environment we control.

## 8. Reproducing

```bash
# the packaging tests, including the ASU isolation guard and the latency report
venv/bin/python -m pytest tests/test_agent_entrypoint.py -q -s

# the competition-shape round robin (needs external/competitors/ populated)
venv/bin/python tools/gauntlet.py --mode rr --candidate LGBM --seeds 3 --workers 10

# the official validator, against a fresh clone rather than the working copy
git clone <repo-url> /tmp/check && venv/bin/python -m submission.validate --local /tmp/check --pretty
```

Validate a **fresh clone**, never the working copy: `--local .` measures the
whole directory, and `venv/`, `external/` and `artifacts/` are gitignored but
still on disk, which reports ~12.6 GB against the 100 MB cap.
