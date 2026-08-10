# CLAUDE.md — Monopoly RL Competition Project

This file exists so that any future session (human or AI) picking up this
project does not re-derive, re-litigate, or accidentally reverse decisions
that have already been checked against the actual code and settled by the
team. Where something is marked **CONFIRMED**, treat it as closed — verified
against source, not just discussed. Where something is marked **OPEN**, it
still needs a decision or a check before the deadline.

---

## 1. Competition context

- 4 teams per match, round-robin. You do not get to see the other three
  teams' agents before the match.
- **5 days total** to finalize a submission. Time, not compute quality, is
  the binding constraint — prefer verified, faithful reproductions of known
  recipes over novel implementations.
- **Submission constraints (CONFIRMED with the team, 2026-08-10):**
  - The submission **must be a learned model**. A pure heuristic or search
    agent is not an acceptable submission.
  - **Imitating ASU is forbidden.** No distillation, no behaviour cloning, no
    use of ASU decisions as supervised targets. ASU is allowed **only as a
    training opponent and as an evaluation benchmark**. No ASU code may run at
    match time either, and nothing the submission imports may reach it.
    - This also rules out `monopoly_bench`'s MonopolyZero bootstrap, which
      trains its policy head on ASU actions (`collect_asu_examples` →
      cross-entropy on `selected_actions`). Its self-play stages are fine; the
      ASU-imitation bootstrap is not.
    - Measured cost of using ASU as an opponent (M4 Pro, 2026-08-10):
      **0.56 s/game vs three fixed agents, 80.8 s/game with one ASU seat,
      54.6 s/game with three.** ASU opposition is ~100× slower, so it must be
      budgeted, not used for bulk training.
  - Hardcoded rules taken from the papers / §4 of this file (the hybrid
    `BUY_PROPERTY` / `ACCEPT_TRADE` split) **are** allowed in the submission.
  - Per-move time budget is unknown; design for ~1 s and measure p95.
- Base codebase: instructor-provided repo, `Darkosxl/DeepRL_Monopoly`
  (https://github.com/Darkosxl/DeepRL_Monopoly). This is itself downstream of
  two academic references (see §6) and reuses their state/action design.

---

## 2. Repository map (confirmed by direct inspection)

> **Read `REPO_STUDY_NOTES.md` before this section.** It is newer and more
> complete than this map was, and it documents the three directories this file
> previously called "unexplored". They are the most developed parts of the repo.

```
DeepRL_Monopoly/
├── ASU_FROZEN_TEACHER/            # frozen ASU-inspired heuristic teachers.
│                                  # asu_value_v1 is the STRONGEST measured
│                                  # policy here (§7). Training-only — §1.
├── RL_CFR_MONOPOLYMODIFIED/
│   └── RL_models_1_CounterfactualRegretMinimization/cfr/classic_cfr.py
├── SLM_HANDMADE_MONOPOLY/         # Gemma 4 QLoRA distillation of ASU labels
├── monopoly_bench/                # MonopolyZero: Max-N PUCT search, self-play,
│                                  # arenas, promotion gates, teacher exports.
│                                  # Owns collect_asu_examples (§8 data path).
├── monopoly_game_engine/
│   ├── actions.py                 # action-space index/offset bookkeeping ONLY
│   ├── env.py                     # game engine + legal-action mask + reward
│   ├── agents_fixed.py            # six scripted personalities, Fixed-A..F
│   ├── networks.py                # ActorNetwork / CriticNetwork / DDQNNetwork
│   ├── constants.py               # referenced, not reviewed line-by-line
│   └── state.py                   # referenced, not reviewed line-by-line
├── tests/
├── tools/
│   ├── train_and_save.py
│   └── play_game.py
├── PPO_PLUS_RULES.md              # ruleset doc — accurate as of d5443fa, see §5
├── README.md                      # points to CFR, PPO/DDQN, and SLM approaches
├── REPO_STUDY_NOTES.md            # the authoritative architecture walkthrough
├── TRAINING_RESULTS.md            # actual measured results, see §7
└── training_guard.py              # RAM guard during training, not an action-mask
```

`training_guard.py` is a memory-safety guard (checkpoints every 100 games,
soft-stops at 3 GiB process RSS, hard-refuses at 4 GiB, stops if
system-available RAM drops to 2 GiB). It has nothing to do with action
legality — don't confuse it with the mask logic below.

---

## 3. The ruleset — CONFIRMED, no code change needed

**Uneven building within a color group is illegal. Uneven selling is legal.**
The even-*build* half is the standard official Monopoly rule and is enforced by
the engine; the selling half is deliberately unenforced (see the end of this
section). Confirmed directly against source, not inferred from documentation.

(The previous version of this headline claimed uneven *selling* was illegal too,
which contradicted this section's own closing paragraph. The closing paragraph
was the correct one.)

The enforcement lives in `monopoly_game_engine/env.py`:

```python
def _is_least_developed(self, prop: Property) -> bool:
    return prop.houses == min(
        self.properties[sq].houses for sq in COLOR_GROUPS[prop.color]
    )
```

This is checked in **two independent places**, and both are correct and
should not be touched:

1. `_improve_actions()` — gates whether `improve_house` / `improve_hotel` for
   a given property appears in the legal-action list at all (mask-generation
   side).
2. `_apply_action()` — re-checks the same condition when the action is
   actually executed (execution-side). If an illegal build somehow made it
   past the mask, this second check silently no-ops it rather than applying
   it.

**Do not remove `and self._is_least_developed(prop)` from either location.**
An earlier pass on this project mistakenly proposed deleting this check,
based on an initial (incorrect) reading of the competition brief as
*granting* uneven building. The team reviewed the code and confirmed: uneven
building must stay illegal. The check is correct as written. If a future
session sees this condition and is tempted to "fix" it — don't. This has
already been investigated once and settled.

Selling is intentionally different: `sell_house` / `sell_hotel` legality
(same file, `_improve_actions()`) does **not** call `_is_least_developed`,
so uneven *selling* is unrestricted. That asymmetry is real and intentional,
not an oversight — confirmed against the same source.

---

## 4. Hybrid agent design — CONFIRMED, already implemented

The action space (`monopoly_game_engine/actions.py`) already marks the
fixed-policy split in the enum itself:

```python
BUY_PROPERTY  = 3   # fixed-policy in hybrid agent
ACCEPT_TRADE  = 7   # fixed-policy in hybrid agent
```

- **Fixed/rule-based:** `BUY_PROPERTY`, `ACCEPT_TRADE`. Low-complexity,
  well-understood decisions — not worth learning.
- **Learned (RL policy):** everything else — mortgage/unmortgage, improve
  house/hotel, sell house/hotel, sell property to bank, all three trade-offer
  types (buy/sell/exchange), auction bidding.

This matches the split validated in the Bonjour et al. paper (§6) and is
already wired in. No changes needed here.

---

## 5. Reward function — CONFIRMED, already implemented

`env.py`, `_compute_reward()`:

```python
own = self.players[pid].net_worth()
mean_other = mean(net_worth of all other active players)
reward = clip((own - mean_other) / (|own| + |mean_other| + 1e-9), -1, 1)
# -1.0 if bankrupt; +1.0 if last player standing
```

This is potential-based net-worth-difference shaping plus a terminal ±1.
This is the correct design (matches the validated approach in the
literature, §6) and should not be changed.

**The documentation bug this section used to flag is FIXED — CLOSED.**
`PPO_PLUS_RULES.md` no longer claims even building is unenforced; commit
`d5443fa` ("Enforce even building and save latest outputs", 2026-08-10)
corrected both that line and the mortgage/building line, and added the positive
statement "Buildings must be distributed evenly across each color group."
The doc now matches §3. Nothing further to do here.

---

## 6. Literature evaluated

| Source | What it actually is | Relevance |
|---|---|---|
| Bonjour, Haliem, et al., "Decision Making in Monopoly using a Hybrid Deep RL Approach," arXiv:2103.00683 / IEEE TETCI 2022 | Full 4-player Monopoly as MDP. 240-dim state, ~2922-dim action space (this repo's 300/2958 dims are a direct descendant). Hybrid PPO vs hybrid DDQN, both vs fixed-policy baselines and each other. | **Primary reference.** Hybrid PPO: 91.65% win rate vs fixed-policy baselines. Hybrid DDQN: 76.91%. Head-to-head, hybrid PPO beat hybrid DDQN 69.06% to 28.56%. |
| Haliem, Bonjour, et al. (earlier preprint, same authors), ResearchGate | DQN warm-started via imitation of a rule-based agent instead of random ε-greedy exploration. | Background only. 23-dim compressed state (group-level, not per-property) — inferior for this project since per-property granularity matters for building decisions. Not what this repo's state design follows. |
| Kejriwal & Thomas, GNOME-p3 simulator (github.com/mayankkejriwal/GNOME-p3) | The simulator both papers above are built on. Built under DARPA's SAIL-ON program specifically to study agents' response to *rule-change novelties* in Monopoly. | Background/lineage only. This repo's engine is a further-modified descendant, not a direct fork. |
| Mnih et al., "Asynchronous Methods for Deep RL" (A3C), arXiv:1602.01783 | Foundational actor-critic paper; ancestor of the GAE-style advantage estimation both Monopoly papers use. | Background only — not implemented directly. PPO (its practical successor) is what's actually used. |
| OpenAI Spinning Up | Educational RL reference/pseudocode. | Reference for hyperparameter sanity-checks and debugging, not a build target. |
| "Monopoly Deal: A Benchmark Environment for Bounded One-Sided Response Games," arXiv:2510.25080 | CFR applied to **Monopoly Deal, the 2-player card game** — not the board game. Different game entirely. | Low relevance. Do not confuse with this project's CFR path (§8), which is a separate, repo-native implementation applied to the actual 4-player board game engine. |

---

## 7. In-repo empirical evidence (from `TRAINING_RESULTS.md`, measured 2026-08-09, RTX 4050 laptop GPU)

- **Hybrid PPO v2, 2,000 games:** final 40-game win-rate window **0.0%**,
  best window across the whole run **2.5%**. ~0.94 s/game, ~1.4M policy
  steps. This is a large, suspicious collapse relative to the paper's
  results at a comparable game count — see §9 for the team's current
  read on why.
- **CFR-style rollout regret matching, one full game:** ~25.5 minutes
  wall-clock for a *single* game. At that rate, CFR cannot produce enough
  games to be viable inside 5 days. Not formal MCCFR — no equilibrium
  guarantee, explicitly noted in the repo's own docs.
- **DDQN, 2,000 games (v2): 0 wins out of 200 evaluation games**, with
  frequent trade/mortgage cycling. Recorded in `REPO_STUDY_NOTES.md` §6, not in
  `TRAINING_RESULTS.md`. This closes the "untested hypothesis" note that used to
  sit here: DDQN was tested, and it is not a credible fallback.
- **`asu_value_v1` (frozen heuristic teacher): 72 wins out of 100**,
  seat-balanced, against Fixed-A/B/C (Fixed-A 0, Fixed-B 10, Fixed-C 18).
  `REPO_STUDY_NOTES.md` §1. **This is the strongest measured policy in the
  repo by a very wide margin** and it is the reason §8 changed direction. It
  is evidence for that exact matchup only, not a universal ranking.

**Measured on the M4 Pro (12 cores, 24 GB, MPS, no CUDA) on 2026-08-10 —
this is the machine that will run the real work, closing the §11 throughput
item:**

| Thing | Measurement |
|---|---|
| `asu_value_v1` decision | 57 ms mean, 324 ms worst |
| `asu_rollout_v1` decision | 12–29 s — unusable at match time, and it burns 11.6 s even when only one action is legal |
| ASU labelled collection | 62 s/game, 379 labels/game (≈10 games/min across 10 workers) |
| PPO training | **0.54 s/game on CPU** — faster than the 0.94 s/game RTX 4050 reference above |

Note the last row: this laptop is not the bottleneck people assumed. A
2,000-game PPO milestone costs ~18 minutes here.

---

## 8. Algorithm decision — PPO, trained by playing. Not by imitation.

> **A distillation plan was proposed and killed on 2026-08-10.** It would have
> trained a network on ASU's decisions. The instructor forbids imitating ASU
> (§1), so it is off the table. Recorded here so nobody re-derives it: the
> reasoning was "ASU wins 72/100, copying it is cheaper than rediscovering it",
> and the answer is that the rule does not allow copying. **Do not re-propose
> distillation, behaviour cloning, or the MonopolyZero ASU bootstrap.**
>
> **Current decision: RL against an opponent curriculum.** Bulk training against
> the fixed agents (fast, 0.56 s/game), with ASU seats as the hard end of the
> curriculum and as the evaluation benchmark — budgeted, because ASU opposition
> costs ~100× more per game (§1).
>
> `train()`, `evaluate()`, `train_ppo()` and `train_ddqn()` now take an
> `opponents` argument naming the three seats, using the same vocabulary as
> `ASU_FROZEN_TEACHER.evaluate` (`fixed-a`..`fixed-f`, `asu-value-v1`,
> `asu-rollout-v1`). It defaults to Fixed-A/B/C, so existing callers are
> unchanged. ASU is imported lazily inside `build_opponents` so that importing
> the engine never drags the teacher into the submitted agent's process.

**Hyperparameters — the paper's exact validated values** (Bonjour et al.
appendix), not whatever produced the 0-2.5% result:

```
γ = 0.9999
GAE λ = 0.95
actor learning rate = 1e-6
critic learning rate = 1e-6
batch size = 5
memory size (rollout length before update) = 20
network: 2 hidden layers, 1024 and 512 units, ReLU
```

**Reasoning trail (why this is PPO and not DDQN), preserved so it isn't
re-argued from scratch:**

1. The only *controlled, direct comparison* evidence available — same
   simulator lineage, same hybrid design, same opponents — is the Bonjour
   et al. paper, and it favors PPO by a wide margin in every configuration
   tested (§6).
2. This project initially followed that evidence, then reversed to DDQN
   after seeing the in-repo 0-2.5% PPO result and the repo's own README
   deprecation note.
3. That reversal was itself flawed: the DDQN recommendation in the README
   was written *in reaction to* the same bad PPO run, not validated by an
   independent DDQN result. Discrediting the PPO run (as poorly trained/
   mistuned) undermines the reasoning for switching to DDQN at the same
   time — they are not independent facts.
4. A 0% → 2.5% collapse is a more typical signature of a mistuned or buggy
   PPO run (PPO is known to be more hyperparameter-sensitive than DQN-family
   methods) than of PPO being structurally unsuited to this action space.
   The paper's own PPO runs, at a comparable game count, won a substantial
   fraction of games.
5. **Decision:** go with PPO using the paper's exact hyperparameters above.
   DDQN's implementation in this repo is also a faithful reproduction of the
   paper's DDQN hyperparameters and remains a credible fallback, but is not
   primary.

**The old OPEN item — "re-run the 2,000-game PPO milestone with the corrected
hyperparameters before committing" — is no longer on the critical path**, because
the plan no longer commits to PPO-from-scratch. It is now optional diagnostic
work: it costs ~18 minutes on this laptop (§7) and would tell us whether the
0-2.5% collapse was mistuning or a real bug in the PPO path (mask wiring, reward
hookup, log-prob computation). That answer still matters for the Phase-4 fine-tune,
so run it if the fine-tune misbehaves — but do not block the distillation on it.

---

## 9. CFR and SLM paths

- **CFR** (`RL_CFR_MONOPOLYMODIFIED/`): ruled out as a primary path. ~25
  minutes per game makes it impossible to generate enough training games in
  5 days. Kept only as a secondary/exploratory item if PPO is solid and time
  remains — not a priority.
- **SLM (small language model) fine-tuning** (`SLM_HANDMADE_MONOPOLY/`):
  mentioned in the repo's own README as a third approach. **Not evaluated
  in this project at all.** Flagged here as a genuine gap — someone should
  at least look at what's in that directory before assuming it's a dead
  end, since it hasn't actually been assessed.

---

## 10. Risk register / guardrails (carry these into training, don't relearn them the hard way)

- **Verify simulator throughput before committing wall-clock budget.**
  ~0.94 s/game was measured for PPO on the reference hardware. Confirm this
  holds on whatever machine will actually run the real training, since it
  directly determines how many games are achievable in the time left.
- **Non-transitivity risk:** an agent tuned only against this repo's
  fixed-policy agents and/or self-play can still lose badly to a genuinely
  different strategy from one of the other three teams, which you cannot
  see in advance. Don't over-trust a single win-rate number from a narrow
  opponent pool.
- **Reward-hacking watch:** the net-worth-ratio-style reward could in
  principle reward a policy that plateaus into a "safe but never wins"
  strategy (excess mortgaging/unmortgaging, cash-hoarding) rather than
  genuinely competing. Watch training curves for this signature, not just
  the final number.
- **Set a go/no-go threshold now, not at the deadline (OPEN — not yet
  decided):** pick a minimum win rate against the strongest fixed-policy
  agent already in this repo that counts as "good enough to submit." If the
  trained model doesn't clear it, submit the tuned fixed-policy agent
  instead of a half-trained model — a mediocre learner that sometimes loses
  to weak fixed play is worse than a heuristic that reliably wins.
- **Opponent diversity in training:** don't train purely via self-play.
  Mix in this repo's existing fixed-policy agents, and consider adding one
  deliberately aggressive/edge-case-seeking opponent profile so the trained
  agent has seen unusual play styles before match day.

---

## 11. Open items checklist (unresolved as of this writing)

- [ ] Re-run the 2,000-game PPO milestone with the corrected hyperparameters
      (§8) and compare against the 0-2.5% baseline.
- [ ] Fix the stale line in `PPO_PLUS_RULES.md` claiming even-building isn't
      enforced (documentation-only fix, §5).
- [ ] Decide and record the go/no-go win-rate threshold (§10).
- [ ] Confirm actual games/sec throughput on the hardware that will run the
      real training run, not just the reference measurement.
- [ ] Take a look at the `SLM_HANDMADE_MONOPOLY/` path — currently
      completely unassessed.
- [ ] Finalize the training opponent mix (fixed agents / self-play /
      adversarial probe agent).
