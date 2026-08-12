# engine/ — the competition simulator, pinned. Do not edit anything in here.

This folder is a **frozen copy** of the organizer's Monopoly simulator, taken from

```
reference/DeepRL_Monopoly_feature/monopoly_game_engine/   @ commit 7b4081b
```

Ten `.py` files, byte-for-byte identical to that commit. `PINNED.json` records the
upstream commit and the SHA-256 of every file.

## Why a copy at all

`reference/` is read-only and is a git worktree that can move under us — a `git pull`, a
branch switch, a stray checkout, and suddenly every number we have ever measured was
measured against a different game. A pinned copy makes the engine a fixed, checkable
thing. `PINNED.json` is the receipt.

## The one rule

**Never edit a file in `engine/`.**

Not to add a print. Not to fix a bug. Not "just for a second to test something."

The moment a vendored file differs from its pin, our results stop being comparable to
the organizer's, and — worse — nobody can tell which past measurement was taken against
which engine. That failure is silent and it is not recoverable after the fact.

## What to do instead

**Need different behaviour?** Write a patch.

Every local change lives as a `.patch` file under `patches/`, and each one must re-apply
cleanly to a *fresh* copy of the upstream files. That way the change is visible, is
reviewable, is reversible, and survives a re-vendor. Add the patch to the
`patches_applied` list in `PINNED.json` and re-run the verifier so the hashes match the
patched state deliberately, not accidentally. See `patches/README.md`.

**Need extra behaviour?** Write a new module in `mono/`.

Most things we want — a seat-invariant observation, an arena that reports both win-rate
conventions, a legality clamp — are *additions*, not edits. They belong in `mono/` and
they should treat `engine/` as a library they call, never as a file they touch. This is
where nearly all our code should go.

## Checking

```bash
OMP_NUM_THREADS=1 .venv/bin/python tools/verify_engine.py             # fast: hashes + ruleset
OMP_NUM_THREADS=1 .venv/bin/python tools/verify_engine.py --upstream  # also re-reads git blobs
```

Exit 0 clean, exit 1 drift. Run it before any training run, before any evaluation you
intend to quote, and before any commit that touches `engine/`. It checks four things: the
pinned files still hash correctly, no unpinned file has appeared, the six
organizer-comparable fingerprints still match, and the live ruleset constants still read
`ppo-plus-v2 / 300 / 2958 / 4`.

Those six fingerprints use the exact file list and key format of
`monopoly_bench/engine.py:202-208` (`engine_hashes()`), so ours are directly comparable
to the organizer's. They were cross-checked against upstream's own function at vendor
time and matched 6/6.

## Importing it

```python
import mono                      # do this first — see below
from engine.env import MonopolyEnv
from engine.agents_fixed import TheHoarder, TheDealMaker, TheGambler
```

Importing `mono` binds our `engine/` to the name `monopoly_game_engine`, which is what
the organizer's `monopoly_bench` imports. Without that bind, using both packages loads
**two** copies of the engine with two distinct `MonopolyEnv` classes, and every
`isinstance` check in the bench quietly takes the wrong branch. Verified: with `import
mono` first, `monopoly_bench` and `engine` share one class object and 8/8 arena games ran
with 0 illegal actions and 0 crashes.

Note `engine/__init__.py` is vendored as-is and hard-imports `torch`, so `import engine`
costs a torch import (~1-2 s). Unavoidable without a patch, and not worth one.

## Re-vendoring

Only when the organizer changes the ruleset. Re-copy from the new commit, re-run the
`engine_hashes()` cross-check, regenerate `PINNED.json`, re-apply every patch in
`patches/`, and **re-measure everything** — a ruleset change invalidates all baselines.
