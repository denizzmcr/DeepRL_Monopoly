# Handoff — state as of 2026-08-11 11:30

Read this before `CLAUDE.md`. Where they disagree, this file is newer.

## Rules we work under

1. The submission must be a **learned model**.
2. **Imitating ASU is forbidden** — no distillation, no behaviour cloning, no ASU
   decisions as supervised targets. This also rules out `monopoly_bench`'s
   MonopolyZero bootstrap, which trains its policy head on ASU actions.
3. ASU **is** allowed as a training opponent and as a benchmark.
4. At most **5 hardcoded rules**. We currently use **3**, listed below.

## The agent

**`artifacts/CHAMPION.pt` is committed to this repo** (normally `artifacts/` is
gitignored; this one file is force-added so both machines have it).

PPO, hybrid, `restrict_liquidation=True`, `entropy_coef=0.005`, 4000 games
trained from scratch against a league of scripted and neural opponents.

Measured, seat-balanced, 80 games per cell (`artifacts/diag/roundrobin2/summary.json`):

| field | champion |
|---|---|
| trained trio (Fixed-A/B/C) | 57.5% |
| unseen trio (Fixed-D/E/F) | 68.8% |
| builders x3 | 97.5% |
| dealmakers x3 | 35.5% (measured over 2500 games; the 31.2% here came from an 80-game cell) |
| strong mix | 67.5% |
| blocker mix | 57.5% |
| vs a rival neural agent | 35.0% |
| vs another rival neural agent | 28.7% |
| **worst case** | **28.7%** |

In a mixed field it beats the strongest scripted agent 52% to 26%. Parity in a
four-player game is 25%.

## The three hardcoded rules

1. `fixed_buy_decision` — buy if it completes a monopoly, else if $100 would remain
2. `fixed_accept_trade_decision` — accept if it completes a monopoly, else if net worth change >= 0
3. `action_filters.restrict_actions` — never voluntarily mortgage or sell; debt-forced liquidation is never blocked

Rule 3 is the one that matters. Without it the same network scores **0%** and
liquidates 251 times per game, because masked actions never received gradient and
their logits are still at initialisation. **A checkpoint trained with the
restriction is wrong to run without it**, which is why the flag is stored inside
the checkpoint and why the evaluator now honours it.

## Findings that are settled — do not re-derive

- **The 0-2.5% collapse was the action space, not tuning.** The paper's own
  hyperparameters reproduce it exactly (2.0% final). Masking voluntary
  liquidation, changing nothing else, gives 76%.
- **Random play wins 0/200 against Fixed-A/B/C. Random play that refuses to
  liquidate wins 18.5%.** That one measurement explains the whole project.
- **Seat 0 is not disadvantaged** — Fixed-D there wins 61%.
- **Non-transitivity is large and real.** Our previous champion scored 36.5% and
  19.0% in two four-ways differing only in the fourth player.
- **Training on all six scripted bots at once destroyed an agent** (12% vs
  Fixed-A/B/C). A rotating league works; a naive mix does not.
- **ASU-only training produced the worst floor of 11 candidates (16.2%).**
- **ASU opponents cost ~100x** — 0.56 s/game scripted, ~55 s/game with ASU seats.

## Machine 1 is currently running

Wave 2 training (10 runs), an ASU continuation run, two ASU evaluations, and a
10,000-game log export. **It is fully saturated.** Machine 2 has the free compute.

**Do not edit** `monopoly_game_engine/train.py` or `agent_ppo.py` — Machine 1 is
training through them. Add new modules instead.

## Machine 2's tasks, in priority order

### 1. Submission packaging (must-have, ~1-2 h)

- **`submission_agent.py`** at repo root: `SubmissionAgent(player_id, checkpoint)`
  with `choose_action(env) -> int`. CPU, `eval()` + `inference_mode()`, hybrid
  interception, masked argmax, fail-closed legality assert. **It must honour
  `restrict_liquidation` from the checkpoint's `training_config`** — running the
  champion without it takes it from 83% to 0%. It must **not** import anything
  from `ASU_FROZEN_TEACHER`; reimplement the inference path locally rather than
  reusing `_NeuralAdapter`.
- **`tests/test_submission_agent.py`**: import `submission_agent` in a fresh
  interpreter and assert no `ASU_FROZEN_TEACHER` module lands in `sys.modules`;
  plus a seeded full game with zero illegal actions and a p95 latency report.
- **`SUBMISSION.md`**: checkpoint SHA-256, the table above, opponent identity,
  seed sets, and the round-cap/truncation counts.

### 2. ANSWERED on 2026-08-11 — do not redo this

The "~31% against Deal-Makers" weakness was measured on 80-game cells. At 2,500
games it is **35.5%**, and parity in a four-player game is 25%, so the champion
is above its share even there. More importantly, the mechanism is not weakness:

| | vs three Deal-Makers | vs Fixed-A/B/C |
|---|---|---|
| games hitting the 200-round cap | **73%** | ~15% |
| mean rounds | **172** | 100 |
| players bankrupt at game end | **0.92** of 4 | 2.75 of 4 |
| largest portfolio anyone holds | 14.6 deeds | 25.7 deeds |

Three aggressive traders keep the board fragmented — nobody assembles a
dominant position, so rents stay low, nobody goes bankrupt, and the game runs to
the cap where the winner is decided on net worth. It is close to a coin flip
among four survivors, and we take 35.5% of it. The league could not "fix" this
because there is nothing to fix; it is a property of that opponent field.

The open follow-up, if anyone wants it: in cap-decided games the objective is
maximum net worth rather than elimination. Whether the agent should play
differently once a game is clearly heading for the cap is untested.

Game records for this matchup are in `artifacts/game_logs/full/dealmakers/`
(2,500 games) and `artifacts/game_logs/full/blockers/` (2,500).

### 3. Parallel trajectory collection — NOW THE SECOND PRIORITY (~1-2 h)

PPO here plays one game at a time in one process, which is why ASU experiments
are prohibitive: 275 games in 10 hours. Workers playing games with the current
weights, returning trajectories to one learner that updates and broadcasts, is
standard PPO practice (the original paper collects from parallel actors) and
would take 2,000 ASU games from ~30 hours to ~3.

Write it as a **new module**, not by editing `train.py`. Reuse
`monopoly_game_engine.train.build_opponents` for opponent construction — it
already accepts `fixed-a`..`fixed-f`, `asu-value-v1`, and `ppo:/path/to.pt`.

## Useful commands

```bash
venv/bin/pip install pytest                     # not installed by default

# seat-balanced evaluation; --focus and --opponents both accept ppo:/path
python -m ASU_FROZEN_TEACHER.evaluate \
  --focus ppo:$PWD/artifacts/CHAMPION.pt \
  --opponents fixed-b fixed-b fixed-b \
  --seeds $(seq 0 49) --output artifacts/diag/champ_vs_dealmakers.json

# full game records, parallel and compressed
python tools/export_game_logs.py --games 200 --seed-base 1 --rotate-seats \
  --workers 8 --compress \
  --players ppo:artifacts/CHAMPION.pt fixed-b fixed-b fixed-b \
  --out-dir artifacts/game_logs/dealmakers
```

Baseline test suite: **119 passed, 9 failed**. All 9 failures are pre-existing and
caused by the missing gitignored `artifacts/ppo_plus/ppo_hybrid_2000_v2.pt`.
`tests/` alone is 97 passed, 0 failed.

## Open questions

- Does wave 2 beat the champion? Machine 1 answers this around midday.
- Can an ASU specialist actually be trained? Two evaluations are running; if yes,
  task 3 makes it affordable and it becomes a league opponent, on the theory that
  rival teams training against ASU will produce similar agents.
- Search at inference (`monopoly_bench/search.py`'s Max-N PUCT over our own
  policy, with a value head trained on our own self-play winners — no ASU) is the
  largest untried gain and the thing rival teams are least likely to attempt.
