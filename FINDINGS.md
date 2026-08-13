# Findings

What we measured, what worked, and what did not. The failures are here on
purpose: three of them changed the direction of the project, and each cost real
compute to establish.

All numbers are seat-balanced — every seed played from all four seats, so seat
advantage cancels. **Parity in a four-player game is 25%.**

---

## 1. The result that defined the project: it was the action space, not the tuning

Hybrid PPO trained to a **0.0% final / 2.5% best** win rate. The obvious
suspicion was hyperparameters. It was not.

| configuration | final window | best window |
|---|---|---|
| code defaults | 0.0% | 2.0% |
| lower entropy coefficient | 2.0% | 10.0% |
| **the paper's exact published hyperparameters** | **2.0%** | **2.0%** |
| defaults + voluntary liquidation masked | 44.0% | 66.0% |
| **low entropy + voluntary liquidation masked** | **76.0%** | **90.0%** |

The paper's own values reproduce the collapse exactly. The single change that
fixes it is refusing voluntary liquidation — mortgaging and selling when not
actually in debt.

The cleanest way to see it: **random play wins 0/200 against Fixed-A/B/C; random
play that merely refuses to liquidate wins 18.5%.** The network is worth about
+35 points on top of that, but the action-space restriction is worth more than
the network.

---

## 2. Field dependence is larger than any difference between agents

The same policies, rescored in tournaments differing in only two of six entrants:

| policy | field A | field B | field C |
|---|---|---|---|
| heuristic | 45.9% | 34.1% | 27.2% |
| ASU | 33.8% | 41.6% | **55.9%** |
| our best learned model | 30.3% | 25.0% | 26.9% |

Changing two entrants flips which of ASU and the heuristic ranks first. A single
win-rate number therefore does not identify the better agent, and we stopped
quoting one. **Selection is on worst case across fields.**

This is also why our learned model was preferred over higher-scoring distilled
networks: 23.8% worst case versus 12.5%.

---

## 3. Distilling a heuristic into a network: works, and still loses

The strongest policy available is a hand-written heuristic. Turning it into a
network is legal, cheap (0.07 ms per decision, so labels cost only the price of
simulating the game), and it converged well:

| stage | result |
|---|---|
| 22,000 games collected | 33M labelled decisions, ~5 min across four hosts |
| behaviour cloning | **92%** held-out action agreement |
| one DAgger round | **+8.1 points** win rate, z = 2.41 |

And yet the students lost to a plain PPO agent in every field containing ASU
(14–21% against 25–27%).

**Why:** the student agrees ~92% with the teacher on *the teacher's* states but
only **73–75%** on *its own* — classic compounding distribution shift. DAgger
closes part of that gap. But the deeper problem is that the student successfully
becomes heuristic-*like*, and ASU specifically exploits heuristic-like play. In a
field of four distilled students, ASU scored **55.9%**. Distillation manufactured
ASU's ideal opponent.

**A hypothesis we tested and refuted.** We suspected the students were weak
against ASU because their training data contained no ASU-generated states. We
rebuilt the dataset with 5% ASU-opponent games (ASU as opponent only — never as a
label source). The students got **worse**, not better: 12.5–17.2% against the
21.6% of the version trained without any ASU states. The gap is not a data gap.

---

## 4. PPO cannot fine-tune a distilled policy

Warm-starting PPO from the distilled checkpoint collapsed it from **28% to 0.4%
in 528 games**, with policy entropy climbing 1.21 → 1.44.

We applied every standard fix at once — critic warm-up with the actor frozen
(the critic was never distilled, so its advantages were pure noise), zero entropy
bonus, and a learning rate of **3e-6**, 100× below default:

| learning rate | actor frozen | ~500 games after unfreezing |
|---|---|---|
| 1e-5 | 40.9% | 0.4% |
| 3e-6 | 45.5% | 1.1% |

Entropy still exploded *with the entropy bonus set to zero*. The cause is
structural: a distilled actor puts ~1.0 probability on one action, so PPO's ratio
term π_new/π_old is violently sensitive, and one negative advantage pushes
probability off the chosen action with nothing anchoring it back. Fixing this
needs a KL penalty toward the distilled policy — an algorithm change, not a
hyperparameter.

Note the same run's useful by-product: with the actor frozen, the distilled
student scored **39–45%** on the training league, above the 33–37% of the PPO
agent. The distillation was good. PPO simply cannot refine it.

---

## 5. Rollout search made the teacher worse

To exceed the teacher's ceiling we added search at *training* time only (search
at inference is not permitted; the artefact would still have been a plain
network). Candidate-restricted rollouts, scored by the engine's own reward:

| configuration | searched decisions/game | win rate |
|---|---|---|
| 2 playouts, horizon 8 | 110 | 21.7% |
| 4 playouts, horizon 15 | — | 12.5% |
| also search when variants agree | most | **1.7%** |

Monotone in the wrong direction: **the more the search intervened, the worse it
played.** With 2–4 playouts the value estimate is dominated by dice variance, so
search substitutes noise for the heuristic's informed judgement. A usable signal
needs 30–100 playouts per candidate, 10–25× the cost, which puts label generation
out of reach.

Measuring this before collecting a single label cost 10 minutes and saved a night.

---

## 6. Smaller notes worth keeping

- **DDQN** measured 0 wins in 200 evaluation games. Not a credible fallback.
- **CFR** takes ~25 minutes per game here. Not viable at any useful scale.
- **Network capacity did not matter.** 256, 512 and 1024-wide students all reached
  ~91% agreement and were statistically indistinguishable in play.
- **The paper's absolute reward lost** to potential-difference shaping on all
  three paired seeds.
- **Training against ASU alone** produced the worst floor of eleven candidates —
  a specialist that lost everywhere else.
- **Stalemates are a field property, not a weakness.** Against Deal-Makers, 73% of
  games hit the 200-round cap and only 0.92 of 4 players go bankrupt.

---

## 7. Engineering lessons that cost us time

- **A host is not storage.** Colab reclaimed four sessions; twice results existed
  only on the host and were lost — two hours of collection, then a 4.7-hour
  training run. Everything now writes incrementally and is pulled on a timer
  (`artifacts/ops/puller.py`).
- **Synchronous round barriers punish slow opponents.** With `games_per_round`
  equal to `workers`, every worker idles behind the one game that runs to the
  200-round cap. Raising it to 3× workers took one agent from 0.11 to 0.29
  games/s.
- **Measure the expensive thing before budgeting around it.** Two ETAs in this
  project were wrong in the same direction because ASU's per-game cost was
  estimated rather than measured.
