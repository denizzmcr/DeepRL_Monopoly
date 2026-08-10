# Handoff — competition agent, state as of 2026-08-11

Read this before `CLAUDE.md`. Where they disagree, this file is newer.

## The rules we are working under

1. The submission must be a **learned model**.
2. **Imitating ASU is forbidden.** No distillation, no behaviour cloning, no ASU
   decisions as supervised targets. This also rules out `monopoly_bench`'s
   MonopolyZero bootstrap, which trains its policy head on ASU actions.
3. ASU **is** allowed as a training opponent and as an evaluation benchmark.
4. Hardcoded rules from the papers are allowed, but the instructor wants the
   agent to be roughly 95% RL, so keep them few and measured.

## What we found (this is the important part)

PPO scored 0-2.5% in every earlier run. The cause was not PPO, not the
hyperparameters, and not the reward.

**The agent could liquidate its own assets at almost any decision point, and
any policy with exploration noise did so until it had nothing left.** Measured
over 200 games, agent in seat 0 against Fixed-A/B/C:

| policy | win rate |
|---|---|
| random over all legal actions | 0.0% |
| random, no trade offers | 0.0% |
| random, no voluntary liquidation | 18.5% |
| PPO, unrestricted, 2000 games | 0.0% |
| PPO, unrestricted, low entropy, 2000 games | 2.0% |
| **PPO, liquidation restricted, low entropy, 2000 games** | **76% training / 83% greedy** |

Supporting facts, all measured, so nobody re-derives them:

- **Seat 0 is not disadvantaged.** Fixed-D in seat 0 wins 61%, Fixed-B 43.5%.
- **Fixed-A never wins a game** against anybody, in any seat.
- **The fixed agents are a cliff, not a ladder.** Random beats three Fixed-A
  opponents 96.5% of the time; adding a single Fixed-B drops it to 0%. There is
  no gradual curriculum available from the scripted agents.
- **Removing the restriction after training does not work.** The same checkpoint
  scores 0% and liquidates 251 times per game, because masked actions never
  received gradient and their logits are still at initialisation. Retraining
  with it removed collapses to 0% and never recovers.
- **ASU opposition costs ~100x.** 0.56 s/game against three fixed agents,
  80.8 s/game with one ASU seat, 54.6 s/game with three. Budget it.

## The agent

`artifacts/diag/ppo_restrict.pt` — PPO, hybrid, `restrict_liquidation=True`,
`entropy_coef=0.005`, 2000 games against Fixed-A/B/C. Seat-balanced through the
official evaluator: **80% over 20 games** (Wilson 58-92%). Larger runs in
`artifacts/diag/eval_*.json`.

`artifacts/diag/ppo_nohybrid.pt` — same but `hybrid=False`, so the network makes
every decision. Reached 77% by game 800, i.e. dropping the hybrid hardcodes
costs nothing. **This is the one to prefer** for the 95%-RL requirement.

Note `artifacts/` is gitignored. Checkpoints must be copied deliberately.

## What is hardcoded, exactly

Measured over 58,726 decision points: 80.8% of real choices are made by the
network, 18.4% by the hardcoded trade-accept rule, 0.9% by the hardcoded buy
rule. With `hybrid=False` both rules disappear and the figure is ~100%.

The remaining restriction, `monopoly_game_engine/action_filters.py`, removes
about 11.8 options from an average menu of 80. It never chooses an action, and
it never blocks debt-forced liquidation — when `env.debt_player` is set, the
engine returns liquidation actions exclusively and they are left alone.

## Machine 2's task

Branch from `feat/asu-shards`. Build:

1. **`submission_agent.py`** at repo root — `SubmissionAgent(player_id, checkpoint)`
   with `choose_action(env) -> int`. CPU, `eval()` + `inference_mode()`, masked
   argmax, fail-closed legality assert. **It must honour `restrict_liquidation`
   from the checkpoint's `training_config`** — running a restricted checkpoint
   unrestricted takes it from 83% to 0%. It must NOT import anything from
   `ASU_FROZEN_TEACHER`; reimplement the inference path locally rather than
   reusing `_NeuralAdapter`.
2. **`tests/test_submission_agent.py`** — import `submission_agent` in a fresh
   interpreter and assert no `ASU_FROZEN_TEACHER` module lands in `sys.modules`;
   plus a seeded full game with zero illegal actions and a p95 latency report.
3. **A parallel seat-balanced evaluation runner.** ASU evaluation is the
   bottleneck at ~81 s/game: 200 games is ~4.5 hours serially but ~30 minutes
   across 10 cores. Fan games out with multiprocessing over disjoint seeds,
   rotate the focus agent through all four seats per seed, report Wilson
   intervals. Reuse `monopoly_game_engine.train.build_opponents`.

Do not touch `monopoly_game_engine/train.py` or `agent_ppo.py` — Machine 1 is
training in those files.

## Environment notes

- `venv/bin/pip install pytest` — pytest is not installed by default.
- Baseline is **119 passed, 9 failed**. All 9 failures are pre-existing and
  caused by the missing gitignored `artifacts/ppo_plus/ppo_hybrid_2000_v2.pt`.
  `tests/` alone is 97 passed, 0 failed.

## Open questions

- Does a legal-action restriction count against the instructor's 95% RL rule?
  If it does, the principled fix is to leave liquidation available but train it
  down with a reward penalty, so the network learns it is bad rather than never
  seeing it. That is a real experiment, not a config flag.
- Is the agent learning to **win** or to **be rich**? Games hitting the
  200-round cap are decided by net worth, so "hoard and stall" is a viable
  exploit that would likely fail against unknown opponents. Not yet checked.
