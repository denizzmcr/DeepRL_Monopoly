# Handoff — state as of 2026-08-12 09:30

Read this before `CLAUDE.md`. Where they disagree, this file is newer.

## Rules

1. The submission must be a **learned model**.
2. **Imitating ASU is forbidden** — no distillation, no ASU decisions as targets.
   ASU is allowed as a training opponent and as a benchmark.
3. At most **5 hardcoded rules**. We use **3**.

## What we have

| artifact | what it is |
|---|---|
| `artifacts/CHAMPION.pt` | the current submission — 256-wide, 4000 games, committed to git |
| `submission_agent.py` | the entry point, ASU-free, fail-closed, 112 tests passing |
| `SUBMISSION.md` | checkpoint identity and measured results |
| `external/kuzey/` | a teammate's hand-written heuristic — **stronger than anything we trained** |

`CHAMPION.pt`, seat-balanced at 2,500 games per field:

| opponents | win rate |
|---|---|
| Fixed-A/B/C | 55.4% |
| Fixed-D/E/F | 61.0% |
| mixed scripted | 57.4% |
| Deal-Makers ×3 | 35.5% |
| vs a rival neural agent | 29.8% |
| **3× ASU** | **~17%** |

Parity in a four-player game is 25%.

## The open question, and it is a big one

**Kuzey's heuristic is claimed to score 0.63 against ASU.** We measured it at
**77.5% against Fixed-A/B/C** (our champion: 55%) at **0.07 ms per decision**
(ASU: 57 ms), zero illegal actions over 40 seat-rotated games. Its bundled
engine files are byte-identical to ours.

If the 0.63 holds under a seat-balanced evaluation, then a hand-written
heuristic beats ASU while our best trained agent gets 17% — and that changes
what we should submit, or at least what we should be training against.

**Nobody has verified it independently yet. That is Machine 2's job.**

## What Machine 1 is doing

Three training runs, all 512 or 1024 wide, from scratch, on a league that
contains Kuzey's heuristic (~31% of fields), ASU, scripted agents, our own past
agents, and deliberately collapsed agents:

| run | games | win rate on its league |
|---|---|---|
| fast (512, no ASU) | 18,000+ | 28.0 → 32.5 → 35.7 |
| asu (512, 19% ASU) | 2,500+ | 22.4 → 23.8 → 30.6 |
| asu (1024) | just started | — |

**Both are still climbing.** Every 256-wide agent we ever trained went flat by
~1,000 games; these are 15× past that and improving. The only change is network
width, which is why the capacity hypothesis is currently the live one.

## Findings that are settled — do not re-derive

- **The 0-2.5% collapse was the action space, not tuning.** The paper's own
  hyperparameters reproduce it exactly. Masking voluntary liquidation gives 76%.
- **Random play wins 0/200 against Fixed-A/B/C; random play that refuses to
  liquidate wins 18.5%.** That single measurement explains the project.
- **The network is worth +35 points** over random choice within the same rules.
- **Non-transitivity is large**: our previous champion scored 36.5% and 19.0% in
  two four-ways differing only in the fourth player.
- **ASU-only training produced the worst floor of 11 candidates.**
- **The paper's absolute reward lost** on all 3 paired seeds (46.1 vs 34.1/35.9).
- **Deal-Makers are a stalemate field, not a weakness**: 73% of those games hit
  the 200-round cap, only 0.92 of 4 players go bankrupt, and we take 35.5% of a
  near-coin-flip.
- **Search at inference is not permitted** — do not build it.

## Machine 2's task

**Verify Kuzey's heuristic. Nothing else until that is answered.**

It needs no checkpoints from Machine 1 — the heuristic and ASU are both in this
repo, so it is fully self-contained.

1. **Kuzey vs 3× ASU**, seat-balanced, 200 games (50 seeds × 4 seats). This is
   the 0.63 claim. Expect ~1 hour; ASU costs ~55 s/game.
2. **Kuzey vs Fixed-A/B/C and vs Fixed-D/E/F**, 200 games each. Cheap. Confirms
   our 77.5% spot-check at proper sample size.
3. **Kuzey vs CHAMPION.pt** at the same table — `--focus` and `--opponents` both
   accept `ppo:/path`, so put both in one four-way with two scripted agents.
4. Write `KUZEY_EVAL.md`: the numbers with Wilson intervals, how many games hit
   the 200-round cap, and a plain recommendation on whether this heuristic
   should be the submission instead of a trained agent.

Use `monopoly_game_engine.train.build_opponents`, which now accepts `kuzey` and
`kuzey-plus` as opponent ids alongside `fixed-a`..`fixed-f`, `asu-value-v1`, and
`ppo:/path/to.pt`.

**Do not edit** `monopoly_game_engine/train.py`, `agent_ppo.py`, or
`parallel_ppo.py` — Machine 1 is training through them.

## Environment notes (Windows)

```
venv\Scripts\python              not venv/bin/python
(0..49)                          not $(seq 0 49)
run commands on separate lines   && is unreliable in PowerShell
```

Baseline: `tests/` is 112 passed, 0 failed. The 9 failures in
`monopoly_bench/tests/` are pre-existing and caused by a missing gitignored
checkpoint.
