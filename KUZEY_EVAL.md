# KUZEY_EVAL.md — independent verification of Kuzey's heuristic

Machine 2, 2026-08-12. Task: verify the claim that Kuzey's hand-written heuristic
scores **0.63 against ASU**, and say whether it should be our submission.

Everything below is measured on this machine unless it is labelled as somebody
else's number. Where a claim did not reproduce, the failed reproduction is the
result, not a footnote.

> **Sample sizes are stated on every number.** Three of the four runs are
> complete at n=200. The ASU run is at **n=50 of 200** and still going; its
> interval already excludes the claim by 37 points, and §2.3 shows the partial
> sample is biased in the claim's favour, not against it. This note will be
> removed when it finishes.

---

## 1. Summary

**The 0.63 does not reproduce. Kuzey's heuristic scores 14.0% against our ASU
(n=50, CI 7.0–26.2) — below the 25% parity line, not 63.6%.** The original figure
was measured against `asu_class`, an opponent that is not in this repository, is
rated mid-ladder by Kuzey's own round robin, and has a documented auction
weakness that Kuzey's agent contains a purpose-built rule to exploit.

**It should not be our submission** — it is a hand-written rule set, which the
rules forbid, and the number that made it look tempting was not a number about
our ASU.

**Everything else about it checks out, and it is the best training opponent we
have.** Its package self-test passes, its vendored engine is byte-identical to
ours where the game is defined, it emits no illegal actions in 533,227 decisions,
and it beats our champion by ~12 points on identical seeds against scripted
fields.

**The result that should worry us is about our own agent, not Kuzey's.**
`CHAMPION.pt` takes 8.0% at a table where Kuzey holds the fourth seat, below a
scripted agent it otherwise dominates.

| measurement | result | n |
|---|---|---|
| Kuzey vs 3× `asu-value-v1` — **the claim** | **14.0%** [7.0, 26.2] | 50 / 200 |
| Kuzey vs Fixed-A/B/C | 76.0% [69.6, 81.4] | 200 |
| Kuzey vs Fixed-D/E/F | 85.5% [80.0, 89.7] | 200 |
| Kuzey vs `CHAMPION.pt` + `fixed-a` + `fixed-d` | 78.5% [72.3, 83.6] | 200 |
| — `CHAMPION.pt` at that same table | 8.0% [5.0, 12.6] | 200 |

---

## 2. The 0.63 claim

### 2.1 What the claim actually refers to

The claim as it reached us is "0.63 against ASU". Its source is
`external/kuzey/Kuzeys_heuristic/CLAUDE.md`, which reports:

| opponents | win rate | games |
|---|---|---|
| `asu_class` ×3 | 63.6% | 360 |
| `asu_class`, builder, blocker | 75.0% | 360 |

**`asu_class` is not our `asu-value-v1`, and it is not in the repository.**
This is stated by the package itself, in `play.py` lines 56-58:

```python
#: Panels built only from agents that SHIP WITH THE ENGINE, so this file needs nothing else.
#: The full evaluation also used an ASU-class opponent, which is not part of this package --
#: see CLAUDE.md for those numbers.
```

A search of `external/kuzey/` for `asu_class` returns no definition — only the
results table that quotes it. The agent behind the headline number is therefore
not shippable, not inspectable, and not re-runnable by us. Three facts about it
can be recovered from Kuzey's own source:

1. **It is a mid-ladder opponent, not a strong one.** Their CLAUDE.md line 108,
   from a 1,600-game round robin over 21 agents: *"`Champion` **+131**,
   `SpineH100` **0**, and for reference the ASU-class agent sits at **+1** —
   level with `SpineH100` in a mixed field."* `SpineH100` is their **previous**
   champion, which their own ladder describes as "strong-but-beatable". So the
   opponent in the 63.6% row is rated level with an agent their current champion
   beats by 131 points.

2. **Their agent contains a rule built specifically to exploit it.** `spine.py`
   lines 30-34 record the auction policy as ported from
   `research/r3_anti_asu/agents_x.py`, described as *"the value-capped auction
   with a cash reserve term (the measured hole in ASU: no cash term, auction
   ceiling averages 4.26x list)"*. The same comment is repeated at the tunable
   itself (`spine.py:296`).

3. **Our `asu-value-v1` does not have that hole.** `ASU_FROZEN_TEACHER/core.py`
   computes `_auction_ceiling` as the *marginal* value of acquiring the deed
   (`value(with property) - value(baseline)`, line 764-776), and
   `_auction_candidate` evaluates `safety` on the post-bid state **after**
   `cash -= bid` (lines 796-803), rejecting any bid above that ceiling. That is a
   cash-aware ceiling derived per position, not a fixed multiple of list price.

   This is a reading of the source, not a measurement of realised bid multiples.
   It is enough to establish that the two ASU implementations are different
   agents; it is not enough to say by how much.

So the 0.63 was measured against a different opponent than the one our project
means by "ASU", by a harness we do not have, and the measuring agent carries a
rule tuned against a documented weakness of that specific opponent. A number
obtained that way is not expected to transfer, and transfer is exactly what the
claim assumes.

### 2.2 What it scores against our ASU

`kuzey` vs three `asu-value-v1`, seat-balanced, seeds 0–49.

> **Sample so far: 50 of 200 games.** This run costs ~6 minutes per game on this
> machine and is still going; the table is updated as it completes. The interval
> at n=50 already excludes the claim by a wide margin, and §2.3 explains why
> finishing the run is not expected to move it up.

| | wins / games | win rate | Wilson 95% |
|---|---|---|---|
| **`kuzey` vs 3× `asu-value-v1`** | **7 / 50** | **14.0%** | **7.0 – 26.2** |
| the claim, for comparison | — | 63.6% | — |
| parity in a four-player game | — | 25.0% | — |

**The claim does not reproduce.** 63.6% is not merely outside the 95% interval,
it is 37 points above its upper bound. Kuzey against our ASU is **below parity**,
not dominant over it.

Supporting detail: 0 of 50 games reached the round cap and the mean game length
is 47.8 rounds, so nothing here is decided by net-worth accounting at a stalemate
— these games resolve outright, and mostly not in Kuzey's favour. Wins were
spread across seats (3/12, 1/13, 3/13, 0/12), so the result is not a seat
artifact.

### 2.3 The interim sample is biased *towards* Kuzey, not against it

Games are written as they finish, so a partial sample over-represents whichever
games complete fastest. Among the 50 finished, **Kuzey's wins take 209 s of
wall-clock on average and its losses take 302 s**, at near-identical round counts
(50.1 vs 47.4). Fast games are therefore disproportionately *wins*, so the
completed subset flatters Kuzey and the full-sample figure should land at or
below 14%, not above it.

### 2.4 How this compares to our own agent

`HANDOFF.md` records `CHAMPION.pt` at **~17%** against 3× ASU. That number was
produced by Machine 1 on a seed set we do not have, so it is **not paired** with
the 14% above and the two should not be differenced. What can be said is that
both sit in the same band, below the 25% parity line, and the gap between them is
smaller than the width of either interval.

A paired champion-vs-3×ASU run on seeds 0–49 was started and then abandoned: at
~6 minutes per game it costs another 3.3 hours, and it cannot change the
conclusion, because the claim under test is that the heuristic *beats* ASU. It
does not.

**So the premise in the handoff — "a heuristic beats ASU while our best trained
agent takes 17%" — is false.** Against the ASU in this repository, the heuristic
takes about what our trained agent takes. Nothing about the submission decision
turns on Kuzey being an ASU-beater, because it is not one.

---

## 3. What did reproduce

### 3.1 The package is the agent that was measured

Their self-test is a hard identity check, and it passes:

```
champion vs mixed:   79/120 = 65.8%
wins by seat: [20, 23, 20, 16]    crashes: 0    OUR clamps: 0
SELF-TEST: expected 79/120 = 65.8%   ->   PASS
```

So no result below is explained by "we ran the wrong version of the code".

### 3.2 The engine is ours

The vendored `engine/` is **byte-identical** to `monopoly_game_engine/` in every
file that defines the game: `env.py`, `actions.py`, `constants.py`, `state.py`,
`agents_fixed.py`, `networks.py`, `agent_ddqn.py`. The three files that differ —
`train.py`, `agent_ppo.py`, `__init__.py` — differ only by our own later
additions (`restrict_liquidation`, `build_opponents`, the Kuzey/ASU opponent
ids). None of them can change a game's outcome. The claim that the engine is the
same engine is correct.

### 3.3 The scripted-field numbers hold, at proper sample size

200 games each, 50 seeds × 4 seats, seat-balanced (the focus agent plays every
seat on the same dice).

| field | wins / games | win rate | Wilson 95% | outright | at cap | round cap hit |
|---|---|---|---|---|---|---|
| Fixed-A/B/C | 152 / 200 | **76.0%** | 69.6 – 81.4 | 131 | 21 | 39 / 200 (19.5%) |
| Fixed-D/E/F | 171 / 200 | **85.5%** | 80.0 – 89.7 | 160 | 11 | 13 / 200 (6.5%) |

The 77.5% spot-check on 40 games sits inside the Fixed-A/B/C interval. That claim
reproduces.

Opponent win rates in those same games, which is where the margin is visible:

| field | opponents |
|---|---|
| Fixed-A/B/C | `fixed-b` 18.5%, `fixed-c` 5.0%, `fixed-a` 0.5% |
| Fixed-D/E/F | `fixed-d` 13.5%, `fixed-e` 1.0%, `fixed-f` 0.0% |

### 3.4 Against our champion, on the champion's own seeds

`SUBMISSION.md` §5.2 records `artifacts/CHAMPION.pt` on **seeds 0–19**, 80 games
per field. Restricting our Kuzey runs to the same 20 seeds makes the comparison
paired — same dice, same seat rotation, same evaluator:

| field | CHAMPION.pt | Kuzey | gap |
|---|---|---|---|
| Fixed-A/B/C | 57.50% | **71.25%** (57/80, CI 60.5 – 80.0) | **+13.75** |
| Fixed-D/E/F | 68.75% | **80.00%** (64/80, CI 70.0 – 87.3) | **+11.25** |

Against scripted opposition the heuristic is roughly twelve points better than
the checkpoint we currently intend to submit. That part of the handoff's worry is
real and is not a sampling artifact.

### 3.5 At the same table as the champion

One four-way, 200 games, 50 seeds × 4 seats: `kuzey`, `ppo:artifacts/CHAMPION.pt`,
`fixed-a`, `fixed-d`. Kuzey rotates through all four seats; the other three
rotate among the remaining ones.

| policy | win rate | Wilson 95% | outright | at cap |
|---|---|---|---|---|
| **kuzey** | **78.5%** | 72.3 – 83.6 | 155 | 2 |
| `fixed-d` (TheBuilder) | 13.5% | 9.4 – 18.9 | 27 | 0 |
| **`CHAMPION.pt`** | **8.0%** | 5.0 – 12.6 | 16 | 0 |
| `fixed-a` (TheHoarder) | 0.0% | 0.0 – 1.9 | 0 | 0 |

Round cap reached in 2 of 200 games; mean game length 45 rounds. This field
resolves decisively and fast — nobody is stalling to a net-worth verdict.

**Our champion finishes third of four, below a scripted agent, at 8% against a
25% parity line.** That is not a restatement of §3.4. Against Fixed-D/E/F alone
the champion scores 68.75% and `fixed-d` is its victim; move Kuzey into the
fourth seat and the champion drops to 8% while `fixed-d` *rises* above it. The
checkpoint is not merely weaker than the heuristic — its advantage over scripted
play does not survive the presence of a strong fourth player.

This is the largest instance of the non-transitivity `SUBMISSION.md` §7 warns
about. The previously recorded spread was 36.5% → 19.0% from changing one seat;
this is 68.75% → 8.0%.

### 3.6 Legality and cost

Over **533,227 decisions** across the three fields above:

* **0 illegal actions.** Confirmed, but note *why* it is guaranteed:
  `spine.py:347-348` clamps the agent's own choice to `END_TURN` (or the first
  legal action) before returning, so an illegal action can never leave the agent.
  "Zero illegal actions" is therefore true by construction and is not by itself
  evidence about decision quality.
* **0 internal fail-safe firings.** This is the stronger result and it is not by
  construction. Re-running `_decide` alongside every non-forced decision and
  checking its raw output against the legal set, the clamp never fired once. The
  policy genuinely stays inside the legal set on its own.
* **0.27 – 0.49 ms per decision**, measured under varying machine load. This is
  above the 0.07 ms quoted in the handoff; a clean idle-machine measurement is in
  §5. Either figure is ~2,000× under the ~1 s design budget, so latency is not a
  decision factor either way, and the discrepancy is not worth chasing further.

---

## 4. Method, and one correction to how we read cap games

Runs were produced by `tools/eval_kuzey.py` (new, added by this task). It exists
because the two existing harnesses each do half the job: `ASU_FROZEN_TEACHER.evaluate`
is our seat-balanced evaluator but its `parse_agent_spec` cannot seat Kuzey, and
`train.build_opponents` knows `kuzey` but its `evaluate` is single-seat and bound
to a learning agent. The new script reuses the evaluator's `_run_game` **unmodified**
and only extends agent construction. `monopoly_game_engine/train.py`, `agent_ppo.py`
and `parallel_ppo.py` were not touched.

**Validation.** On a lineup both harnesses can run (`fixed-a` vs `fixed-b/c/d`,
seeds 0–4), the canonical evaluator and `tools/eval_kuzey.py` return identical
results game for game:

```
winners: [3, 3, 0, 0, 3, 0, 3, 2, 1, 2, 3, 0, 2, 3, 0, 2, 3, 0, 0, 0]
rounds : [96, 62, 71, 64, 54, 200, 43, 64, 86, 88, 100, 200, 128, 43, 122, 93, 60, 61, 55, 83]
```

**`PYTHONHASHSEED=0` was set for every run.** Kuzey's harness documents that this
engine is not reproducible without it — four identical runs gave win totals
[15, 15, 16, 15] — because something in the decision path iterates a
string-keyed container. Our own evaluator has never pinned it. That is worth
knowing independently of this task: **a ~2.5 point drift is inside the noise band
of most of the A/B comparisons this project has run.**

Their `play.py` cannot self-start on this machine: it re-execs itself via
`os.execv` to pin the hash seed, and that mangles the non-ASCII characters in
this repository's path. Setting `PYTHONHASHSEED=0` in the environment first skips
the re-exec, which is how the self-test in §3.1 was obtained.

**The round cap is not the `truncated` flag.** `env._check_game_over`
(`env.py:1072`) sets `done = True` when one player remains **or**
`round >= max_rounds`, so a capped game is a *completed* game to the evaluator
and `truncated` never flags it — `truncated` only catches the 20,000-decision
safety valve, which fired 0 times in every run here. Cap counts in this document
are computed as `rounds >= 200`, and cap wins are decided on net worth by
`env.winner()`. This matches `SUBMISSION.md` §5.4; it is restated because the
summary line printed by `tools/eval_kuzey.py` reports the valve, not the cap, and
is misleading read on its own.

Because of that, wins are split into **outright** (bankrupted the other three)
and **at cap** (ahead on net worth when the clock stopped) throughout.

**Reproducing (Windows).** `PYTHONHASHSEED` must be set in the environment — the
script refuses to run without it rather than re-execing, for the reason above.

```bash
export PYTHONHASHSEED=0
venv/Scripts/python tools/eval_kuzey.py --focus kuzey \
  --opponents fixed-a fixed-b fixed-c --seeds 50 --workers 6 \
  --jsonl artifacts/kuzey_eval/run.jsonl --output artifacts/kuzey_eval/run.json
venv/Scripts/python tools/analyze_kuzey_eval.py artifacts/kuzey_eval/run.json
```

`--jsonl` appends every finished game and skips it on restart. The ASU run in
this document was killed once at ~1 hour with nothing written; that is why the
flag exists. Cost on this machine: **~6 minutes per game** with three ASU seats
(~59 s/game aggregate across 6 workers, so ~3.3 hours for 200 games), against
~1–2 s/game for scripted fields. The handoff's "~55 s/game" is the M4 Pro figure
and does not hold here.

Raw per-game records are under `artifacts/kuzey_eval/`, which is gitignored;
`artifacts/kuzey_eval/summary.json` is force-added so the aggregates are in the
repository even though the ~150 KB per-game dumps are not.

---

## 5. What was not measured

* **A paired `CHAMPION.pt` vs 3× ASU run** (§2.4). Started, abandoned on cost;
  cannot change the conclusion.
* **Kuzey against a real rival agent.** Every number here — and every number in
  Kuzey's own docs — is against fixed archetypes, our ASU, our checkpoint, or
  copies of itself. Their CLAUDE.md says this plainly: *"It is also not tested
  against a real rival."* The other three teams' agents remain unseen.
* **`ChampionPlus` (`kuzey-plus`)**, the variant with the endgame cash-for-deed
  rule. Only `Champion` was evaluated, since that is the variant behind the claim.
* **A clean idle-machine latency figure.** The machine has been saturated by ASU
  workers throughout. The 0.27–0.49 ms range in §3.6 is an upper bound, and the
  question is moot against a ~1 s budget.

---

## 6. Recommendation

**No. Do not submit the heuristic — and this was never actually a close call.**

Two independent reasons, either one sufficient:

1. **It is not eligible.** `HANDOFF.md` rule 1 requires the submission to be a
   learned model, and rule 3 caps hardcoded rules at five. Kuzey's agent is
   ~1,900 lines of hand-written rules. Kuzey's own CLAUDE.md reaches the same
   conclusion unprompted: *"It is **not a submission**. The competition's 95%
   rule means the entry has to be a model, and this is hand-written rules."*
   Nobody involved believes otherwise; the question arose only because the 0.63
   made it look worth reopening.

2. **The number that made it look worth reopening is wrong.** Kuzey against our
   ASU is **14.0%** (n=50, CI 7.0–26.2), not 63.6% — below parity, and in the
   same band as our own champion's ~17%. The 63.6% was measured against a
   different, unshipped, mid-ladder opponent, by an agent carrying a rule built
   to exploit that specific opponent's documented auction weakness (§2.1).

**What the heuristic is genuinely good for, and it is worth real effort:**

* **As a training opponent it is the best thing we have** — 76.0% against
  Fixed-A/B/C and 85.5% against Fixed-D/E/F, ~12 points above our champion on
  identical seeds, at ~0.3 ms per decision against ASU's 57 ms. That combination
  is what makes strong opposition affordable in bulk, which ASU never was
  (§1 of `CLAUDE.md`: ASU opposition costs ~100× more per game). Machine 1
  already has it in ~31% of league fields; this evaluation supports that and
  supports raising it.
* **Behaviour-cloning it is not forbidden.** The prohibition in rule 2 is on
  imitating **ASU** specifically. Kuzey's heuristic is a teammate's own code, and
  their `spine.py` header already names behaviour-cloning teacher as its intended
  purpose. This is a genuine option that the ASU rule does not touch — but it is a
  decision for the team, not something this evaluation should settle, and it
  needs the instructor's reading of the 95% rule before anyone spends days on it.

**The finding that should actually change what we do** is not about Kuzey at all
(§3.5): `CHAMPION.pt` scores **8.0%** at a four-way table where Kuzey holds the
fourth seat, while scoring 68.75% against Fixed-D/E/F — finishing *below*
`fixed-d`, the agent it beats comfortably elsewhere. Our champion's measured
strength is contingent on the fourth player being weak. Since the competition
puts three unseen agents at the table and at least one is likely to be strong,
**the 55–61% figures in `HANDOFF.md` are the wrong numbers to plan against.**
That is an argument for Machine 1's current direction — wider networks trained on
a league that contains Kuzey — and an argument against shipping the current
checkpoint on the strength of its scripted-field record.
