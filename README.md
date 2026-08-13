# Monopoly Competition Agent

Four-player Monopoly on the `ppo-plus-v2` ruleset: 300-dimensional observation,
2,958-action space, 200-round cap. This repository holds the submitted agent, the
agents it was selected over, and the measurements that decided between them.

**Start here:** [`SUBMISSION.md`](SUBMISSION.md) — what we submit and why.
**Then:** [`FINDINGS.md`](FINDINGS.md) — what we learned, including what failed.

---

## The submission

**`UNDERDOG`** — a hand-written heuristic, in `external/kuzey/`.

```python
from underdog_agent import Underdog

agent = Underdog(player_id=seat)
action = agent.choose_action(env)          # -> one legal action index
```

`underdog_agent.py` at the repository root is the only file the harness needs.
It wraps the `ChampionPlus` variant, which we selected by measurement rather than
default: it beat `Champion` in **three independent** seat-balanced tournaments
(+5.5, +4.6, +5.3 points; pooled 40.3% vs 35.3%, z = 1.69). It fails closed —
an illegal action raises rather than silently substituting.

Self-check, which reproduces the number it was measured at and prints PASS/FAIL:

```bash
python external/kuzey/Kuzeys_heuristic/play.py
```

### Compliance

| requirement | status |
|---|---|
| does not imitate ASU | **verified** — no ASU import at runtime, no ASU code bundled, and the package contains **no weights, tables or label data of any kind** in which ASU decisions could be stored |
| engine not tampered with | **verified** — the vendored `env.py`, `actions.py`, `constants.py`, `state.py` are byte-identical to ours |
| ASU used only as opponent/benchmark | yes — it appears in measurement tables, never in the agent |

---

## Results

Seat-balanced throughout: every seed is played from all four seats, so seat
advantage cancels. Parity in a four-player game is **25%**.

| agent | pooled win rate | games |
|---|---|---|
| **UNDERDOG (ChampionPlus)** | **40.3%** | 536 |
| the same heuristic's default variant (Champion) | 35.3% | 536 |
| ASU (`asu_value_v1`, benchmark) | 38.6% | 2,480 |
| UNDERDOG vs Fixed-A/B/C, seat-balanced | **70.0%** | 40 |
| `LAST_RESORT.pt` (our best learned model) | ~25% | 2,240 |
| best distilled network | ~21% | 960 |

`LAST_RESORT.pt` is a 512-wide PPO policy trained for 47,500 games. It is kept as
a fallback and is fully packaged: see [`SUBMISSION.md`](SUBMISSION.md).

---

## Reproducing any number here

Every tournament is a script plus its raw results, in `artifacts/ops/evals/`:

```bash
venv/bin/python artifacts/ops/evals/standings.py     # full round robin
venv/bin/python artifacts/ops/evals/matched.py       # matched training volume
```

Each enumerates every 4-subset of its policy pool, plays each table over N seeds
× 4 seat rotations, and reports Wilson intervals. The `.json` beside each script
is the raw game-by-game output.

---

## Repository map

```
underdog_agent.py            THE SUBMISSION -- Underdog(seat).choose_action(env)
submission_agent.py          learned fallback only, clearly labelled as such
artifacts/
  LAST_RESORT.pt             the learned fallback (512-wide, 47.5k games)
  ops/evals/                 every tournament script + its raw results
  ops/                       Colab fleet automation (puller, supervisor)
external/kuzey/              the submitted heuristic
monopoly_game_engine/        game engine, action space, reward, agents
tools/
  distill_kuzey.py           heuristic -> network distillation (collect/dagger/train)
  distill_pipeline.py        unattended end-to-end distillation driver
  search_teacher.py          rollout-search teacher (measured, rejected)
tests/                       112 tests, incl. the ASU-import guard
docs/                        ruleset, architecture notes, measured history
```

**Which file is the submission?** `underdog_agent.py`. `submission_agent.py` is a
second, fully packaged agent kept only as a fallback in case a hand-written
algorithm is not acceptable; its first line says so.

Background reading lives in [`docs/`](docs/): `PPO_PLUS_RULES.md` (the ruleset),
`REPO_STUDY_NOTES.md` (architecture walkthrough), `TRAINING_RESULTS.md` and
`HANDOFF.md` (measured history and operational state), `COLAB_SETUP.md`.

---

## Method, in one paragraph

We trained PPO agents, distilled a heuristic into networks, and tried rollout
search — then selected between them by **worst case across tournaments, not best
case**. That criterion mattered more than any single result: changing two of six
policies in a four-player tournament flipped which of ASU and the heuristic
ranked first. Against three unknown opponents, an agent that is never bad beats
one that is sometimes best. The full reasoning, including three approaches that
did not work and why, is in [`FINDINGS.md`](FINDINGS.md).
