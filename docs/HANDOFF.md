# Handoff — state as of 2026-08-12 17:30

Read this before `CLAUDE.md`. Where they disagree, this file is newer.

## Submit `champion_v2_h512_fast.pt` unless something beats it

`artifacts/diag/champion_v2/champion_v2_h512_fast.pt` — 512-wide, 47,500 games on
the fast league. It is the most robust policy we have measured: **23.8–32.2%
across five different four-player fields**, where every alternative swings much
harder. Parity is 25%.

## the heuristic distillation: done, measured, and it did not win

The instructor approved distilling the heuristic (it is the team's own heuristic; the ban
is on ASU only). The whole pipeline is `tools/distill_heuristic.py` — `collect`,
`dagger`, `train`. It worked exactly as designed and still lost:

| | measured |
|---|---|
| collection | 22,000 games / 33M labels in ~5 min across 4 hosts |
| agreement with the heuristic (held out) | **92%** |
| agreement on the student's OWN states | **73%** |
| DAgger round 1 | **+8.1 points**, z=2.41 |
| best student vs `fast` in ASU fields | **14–21% vs 25–27%** |

That 92% → 73% collapse is the whole story: behaviour cloning is scored on the
teacher's state distribution but plays in its own. DAgger closes part of it.

**The reason to stop, though, is not the gap — it is what the students became.**
They play like the heuristic, and the heuristic-like agents are exploitable by ASU. In a field of
four distilled students, ASU scored **55.9%** while every student sat at 14–18%.
Do not spend more time distilling the heuristic without a plan for that.

## Field dependence is the dominant effect — bigger than any model difference

The same policies, rescored in different four-player fields:

| policy | field A | field B | field C |
|---|---|---|---|
| heuristic | 45.9% | 34.1% | 27.2% |
| asu | 33.8% | 41.6% | **55.9%** |
| fast | 30.3% | 25.0% | 26.9% |
| distilled 1024 | — | 18.8% | 16.9% |

Changing two of six policies flips which of ASU and the heuristic is stronger. **No single
win-rate number identifies the best agent — select on worst case across fields.**
This also retracts an earlier claim in this file: the heuristic is *not* simply stronger
than ASU; it depends entirely on who else is at the table.

## PPO fine-tuning from a distilled checkpoint destroys it — and why

First attempt fell 28% → 0.4% in 528 games, entropy climbing 1.21 → 1.31. Causes,
in order:

1. **Only the actor was distilled.** The critic was random, so every advantage
   was noise. Warm the critic up with the actor frozen before joint training.
2. `lr` 3e-4 is the from-scratch default, far too large for a sharp policy.
3. Any entropy bonus pushes a low-entropy distilled policy back toward uniform.
   Use `entropy_coef=0.0`, not 0.005.

**The corrected version was then tested, twice, and it also collapsed. Do not
re-attempt this without changing the algorithm.**

Critic warm-up (1,500 games, actor frozen), `entropy_coef=0.0`, and lr as low as
**3e-6** — all three fixes applied together:

| | actor frozen | ~500 games after unfreezing |
|---|---|---|
| lr 1e-5 | 40.9% | **0.4%** |
| lr 3e-6 | 45.5% | **1.1%** |

Entropy went 0.12 → 1.44 in both, *with the entropy bonus set to zero*. That rules
out all three hypothesised causes and points at the real one: **PPO's clipped
objective is unstable on a saturated policy.** A distilled actor puts ~1.0 on one
action, so the ratio π_new/π_old is violently sensitive, and a single negative
advantage pushes probability off the chosen action with nothing anchoring it
back. It diffuses to uniform and never recovers.

Fixing it properly needs a KL penalty toward the distilled policy (or a much
tighter clip range) — an algorithm change, not a hyperparameter. Out of scope
this close to the deadline.

One contributing inconsistency worth fixing if anyone retries: the student was
distilled with `restrict_liquidation=False` but fine-tuned with `True`, so it was
asked to act under a different action mask than it learned under.

## What the fine-tune runs did reveal

With the actor frozen, the distilled student scores **39–45% on the fast league**,
where `fast@47k` itself scores ~33–37%. So the distilled model is genuinely
strong — on fields without ASU. Against ASU it is the weakest thing we have
(14–21%). That is the same field-dependence result stated above, and it is the
whole basis for preferring `fast@47k`: not that it is stronger on average, but
that its worst case is much better.

## Colab operational facts

- Sessions were reclaimed after ~7 hours. Pull checkpoints continuously; do not
  leave anything only on a host.
- `v6e1` caps at 3 concurrent; `v5e1` is a separate quota.
- **Concurrent `colab exec` calls race on `sessions.json`** (each writes
  `last_execution`) and can truncate it, after which every `-s c1` silently stops
  resolving and hosts get renamed to defaults. `artifacts/ops/colab_refresh.py`
  now keeps an endpoint→name map in its own file and writes atomically. Prefer
  sequential `colab exec`.
- `games_per_round` must be ~3× `workers` for ASU-league runs (0.11 → 0.29
  games/s measured). Irrelevant for update-bound runs like the fast league.

## What to do next, in order

1. Re-run the corrected fine-tune (critic warm-up / lr 1e-5 / entropy 0) from
   both `fast@47k` and a distilled student. This is the only untested upside.
2. Continue `fast@47k` on the fast league — it plateaued at ~33% on its own
   league at 47.5k games, and its gains historically arrived late (15k → 47.5k).
3. Screen every candidate on **worst case across fields**, not on a single
   tournament. Harnesses: `standings.py` / `screen.py` in the session scratchpad,
   copied patterns are cheap to rebuild.
4. Only then package: `submission_agent.py`, legality + p95 latency, SHA-256 in
   `SUBMISSION.md`.

## Unattended training is running right now

Ten PPO agents on four Colab TPU hosts (the host CPUs, not the TPUs), all on the
ASU league, plus a fast league on the laptop.

| host | accel | cores | agents | workers | games/update |
|---|---|---|---|---|---|
| c1 | v6e1 | 44 | 3 × 512 | 14 | 42 |
| c2 | v6e1 | 44 | 3 × 1024 | 14 | 42 |
| c3 | v5e1 | 24 | 2 × 512 | 11 | 33 |
| c4 | v5e1 | 24 | 2 × 1024 | 11 | 33 |
| laptop | — | 12 | fast league, game 36k, 37% | 10 | 20 |

**`artifacts/ops/supervisor.py` runs the fleet unattended.** Every 15 min it
refreshes the Colab proxy tokens, pulls new checkpoints to
`artifacts/diag/champion_v2/`, restarts any agent whose process died (resuming
from that agent's own checkpoint), and restarts `caffeinate` if it is gone. It
reports but does not repair a whole host going unreachable — that needs an
accelerator assignment a human should decide on. Restart it with
`nohup python3 artifacts/ops/supervisor.py >> supervisor.log 2>&1 &`.

**`games_per_round` must be ~3× `workers` for the ASU league.** Setting it equal
to `workers` puts every worker at the round barrier behind the one game that runs
to the 200-round cap. Measured on the same agent: 0.11 → 0.29 games/s. This does
not apply to the laptop's fast league, which is update-bound (`collect 1.9s
update 7.4s`) and has no straggler to amortise.

**The laptop must stay awake with the lid open.** Sleep kills the supervisor, the
four keep-alive daemons, and the laptop's own run; macOS sleeps on lid-close
regardless of `caffeinate`.

Colab quotas, measured: **v6e1 caps at 3 concurrent**, v5e1 is a separate quota,
plain CPU runtimes another (but only 2 cores, not worth it).

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
| `underdog/` | a teammate's hand-written heuristic — **stronger than anything we trained** |

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

**the hand-written heuristic is claimed to score 0.63 against ASU.** We measured it at
**77.5% against Fixed-A/B/C** (our champion: 55%) at **0.07 ms per decision**
(ASU: 57 ms), zero illegal actions over 40 seat-rotated games. Its bundled
engine files are byte-identical to ours.

If the 0.63 holds under a seat-balanced evaluation, then a hand-written
heuristic beats ASU while our best trained agent gets 17% — and that changes
what we should submit, or at least what we should be training against.

**Nobody has verified it independently yet. That is Machine 2's job.**

## What Machine 1 is doing

Three training runs, all 512 or 1024 wide, from scratch, on a league that
contains the hand-written heuristic (~31% of fields), ASU, scripted agents, our own past
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

**Verify the hand-written heuristic. Nothing else until that is answered.**

It needs no checkpoints from Machine 1 — the heuristic and ASU are both in this
repo, so it is fully self-contained.

1. **the heuristic vs 3× ASU**, seat-balanced, 200 games (50 seeds × 4 seats). This is
   the 0.63 claim. Expect ~1 hour; ASU costs ~55 s/game.
2. **the heuristic vs Fixed-A/B/C and vs Fixed-D/E/F**, 200 games each. Cheap. Confirms
   our 77.5% spot-check at proper sample size.
3. **the heuristic vs CHAMPION.pt** at the same table — `--focus` and `--opponents` both
   accept `ppo:/path`, so put both in one four-way with two scripted agents.
4. Write `KUZEY_EVAL.md`: the numbers with Wilson intervals, how many games hit
   the 200-round cap, and a plain recommendation on whether this heuristic
   should be the submission instead of a trained agent.

Use `monopoly_game_engine.train.build_opponents`, which now accepts `heuristic` and
`heuristic-plus` as opponent ids alongside `fixed-a`..`fixed-f`, `asu-value-v1`, and
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
