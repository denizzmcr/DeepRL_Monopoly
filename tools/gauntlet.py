"""Play UNDERDOG against the other teams' submitted agents.

Six other repositories were published for this competition. This runs our
submission against all of them on one shared engine, seat-balanced, so the
result is a measurement rather than an impression.

Why one shared engine, and why that is fair
-------------------------------------------
Every one of these repositories is a fork of the same base, and four of them
vendor their own ``monopoly_game_engine``. Letting each agent import its own
copy is not an option -- Python binds one module name once per process, so
whichever agent imports first silently decides the rules for everybody. We hit
exactly that failure earlier in this project.

So the rules are pinned to *our* engine, and that is defensible because the
parts that define the game are byte-identical across all seven repositories:
``state.py`` and ``actions.py`` diff to zero lines everywhere. The only engine
edit anyone made is EnzeCbe's ``_compute_reward`` liquidity-risk term, which is
reward shaping consumed during *training* and never read while playing. No
agent is handed different rules from any other.

``_pin_engine`` imports our engine into ``sys.modules`` *before* any competitor
module is loaded. Import caching then wins over ``sys.path``, so a competitor's
``sys.path.insert(0, their_root)`` -- EnzeCbe's ``agent.py`` does precisely
this -- cannot swap the engine out from under a game already in progress.

Calling conventions
-------------------
There is no single agreed interface. Five agents take ``choose_action(env)``,
EnzeCbe takes ``choose_action(state, allowed_actions, env)``, and 6c0de accepts
almost anything. Each adapter below records the shape its author actually
wrote, so nobody is penalised for a harness mismatch.

Illegal moves are counted, not crashed on, and replaced the way the rest of
this repository replaces them (``END_TURN`` when legal, else the first legal
action). An agent that forfeits on an exception would score zero for a reason
that has nothing to do with how well it plays.

Usage
-----
    python tools/gauntlet.py --mode h2h    --seeds 25 --workers 10
    python tools/gauntlet.py --mode melee  --seeds 12 --workers 10
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMP = REPO / "external" / "competitors"

NUM_PLAYERS = 4
MAX_ROUNDS = 200
DECISION_CAP = 24000


# ---------------------------------------------------------------- engine pin

def _pin_engine() -> None:
    """Bind the rules to our engine before any competitor module exists.

    Import the submodules too: a competitor doing ``from
    monopoly_game_engine.actions import ...`` resolves through the parent
    package's ``__path__``, and pinning the parent alone leaves that resolution
    dependent on import order.
    """
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))
    for name in ("monopoly_game_engine",
                 "monopoly_game_engine.actions",
                 "monopoly_game_engine.constants",
                 "monopoly_game_engine.env",
                 "monopoly_game_engine.state",
                 "monopoly_game_engine.networks",
                 "monopoly_game_engine.agent_ppo",
                 "ASU_FROZEN_TEACHER"):
        importlib.import_module(name)


def _load_file(path: Path, modname: str, roots: list[Path]):
    """Load one competitor module under a private name.

    Three repositories name their entry point ``agent.py``. Importing them by
    that name would give all three whichever module loaded first, so each gets
    a unique key in ``sys.modules``.
    """
    if modname in sys.modules:
        return sys.modules[modname]
    added = [str(r) for r in roots if str(r) not in sys.path]
    for p in added:
        sys.path.insert(0, p)
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        sys.modules.pop(modname, None)
        raise
    finally:
        # Our root must stay first: a competitor may have inserted its own.
        for p in added:
            if p in sys.path:
                sys.path.remove(p)
        if str(REPO) in sys.path:
            sys.path.remove(str(REPO))
        sys.path.insert(0, str(REPO))


def _load_pkg(modname: str, roots: list[Path]):
    """Import a competitor package by its real name, for relative imports.

    Used where a module says ``from .board import ...`` and therefore cannot be
    loaded under a private alias. Safe only because the package names involved
    are unique across the seven repositories.
    """
    if modname in sys.modules:
        return sys.modules[modname]
    added = [str(r) for r in roots if str(r) not in sys.path]
    for p in added:
        sys.path.insert(0, p)
    try:
        return importlib.import_module(modname)
    finally:
        for p in added:
            if p in sys.path:
                sys.path.remove(p)
        if str(REPO) in sys.path:
            sys.path.remove(str(REPO))
        sys.path.insert(0, str(REPO))


# ------------------------------------------------------------ agent builders
#
# One builder per team. Each returns an object exposing ``choose_action``; the
# ``conv`` field records which call shape that object expects.

def _b_underdog(seat: int):
    from underdog_agent import Underdog
    return Underdog(seat)


def _b_last_resort(seat: int):
    """Our learned fallback, for the swap question the round-robin raises.

    UNDERDOG finishes below parity in a mixed field while winning most
    head-to-heads, so the useful question is whether the other agent we
    already have does better against the same opposition. Same tables, same
    seeds, so the comparison is paired.
    """
    from submission_agent import SubmissionAgent
    return SubmissionAgent(seat)


def _b_champion(seat: int):
    """The base heuristic variant. UNDERDOG is the *plus* variant."""
    from monopoly_game_engine.train import build_opponents
    return build_opponents(["heuristic"], [seat])[0]


def _b_variant(name: str):
    """Any variant in the heuristic's own registry, as a swap candidate.

    The package ships six composed variants and the submitted one is
    ``st_chest_scrap``. They were screened against our own training league,
    not against this field, so which one is best here is an open question the
    round-robin can answer directly.
    """
    def build(seat: int):
        import sys as _s
        h = str(REPO / "underdog")
        if h not in _s.path:
            _s.path.insert(0, h)
        from heuristic.champion import VARIANTS

        class _W:
            def __init__(self, pid):
                self.player_id = pid
                self._a = VARIANTS[name]()

            def choose_action(self, env):
                return int(self._a.choose_action(env, self.player_id, 0))
        return _W(seat)
    return build


def _b_v2(name: str):
    """Candidates from tools/variants_v2.py, aimed at the measured weakness."""
    def build(seat: int):
        sys.path.insert(0, str(REPO / "tools"))
        from variants_v2 import VARIANTS_V2

        class _W:
            def __init__(self, pid):
                self.player_id = pid
                self._a = VARIANTS_V2[name]()

            def choose_action(self, env):
                return int(self._a.choose_action(env, self.player_id, 0))
        return _W(seat)
    return build


def _b_6c0de(seat: int):
    root = COMP / "6c0de_exposure-monopoly-agent"
    mod = _load_file(root / "agent.py", "team_6c0de_agent", [root])
    return mod.Agent(player_id=seat)


def _b_expo(seat: int):
    root = COMP / "emingurbuz9483_exposure-monopoly-algorithm"
    mod = _load_file(root / "agent.py", "team_expo_agent", [root])
    return mod.MonopolyAgent(player_id=seat)


def _b_boom(seat: int):
    """EnzeCbe's hybrid PPO, rebound to the seat it actually plays.

    Their checkpoint stores ``player_id: 0``, and ``PPOAgent.load`` refuses any
    checkpoint whose stored seat differs from the agent's -- a check their fork
    shares with ours, so their agent loads at seat 0 and nowhere else. Scoring
    them zero for that would measure a metadata check, not a policy: an actor
    network is seat-agnostic here, because the seat enters only through the
    state vector and the legal-action set, and this harness passes both
    explicitly. So load at seat 0, then rebind the seat.
    """
    root = COMP / "EnzeCbe_monopoly-boom"
    mod = _load_file(root / "agent.py", "team_boom_agent", [root])
    agent = mod.Agent(player_id=0)
    agent.player_id = seat
    agent._agent.player_id = seat
    return agent


def _b_inncenta(seat: int):
    root = COMP / "Inncenta_monopoly"
    mod = _load_file(root / "final_agent.py", "team_inncenta_agent", [root])
    return mod.Agent(player_id=seat)


def _b_aline(seat: int):
    root = COMP / "alinebidal10-afk_monopoly-competition-agent"
    mod = _load_file(root / "competition_agent" / "final_agent.py",
                     "team_aline_agent", [root])
    return mod.FinalAgent(seat)


def _b_slayer(seat: int):
    # ``ASU_SLAYER.policy`` imports its siblings relatively, so it has to be
    # imported as a package rather than by file path. The package name is
    # unique across the seven repositories, so this cannot collide.
    root = COMP / "emirkaanozdemr_monopoly" / "DeepRL_Monopoly"
    mod = _load_pkg("ASU_SLAYER.policy", [root])
    return mod.SlayerV1(seat)


# name -> (builder, calling convention, owner)
AGENTS: dict[str, tuple] = {
    "UNDERDOG":  (_b_underdog, "env", "ours"),
    "LAST_RESORT": (_b_last_resort, "env", "ours (learned fallback)"),
    "champion":  (_b_champion, "env", "ours (base heuristic variant)"),
    "core":      (_b_variant("st_core"), "env", "ours (ChampionPlus + thaw)"),
    "stall":     (_b_variant("st_all"), "env", "ours (core + reach)"),
    "thaw":      (_b_variant("st_chest_thaw"), "env", "ours (Champion + thaw)"),
    "v2_plain":  (_b_v2("v2_plain"), "env", "ours (auction ceiling 1.30x)"),
    "v2_plain2": (_b_v2("v2_plain2"), "env", "ours (auction ceiling 1.75x)"),
    "v2_buy":    (_b_v2("v2_buy"), "env", "ours (buy deeper into reserve)"),
    "v2_plainbuy": (_b_v2("v2_plainbuy"), "env", "ours (both)"),
    "v2_horizon": (_b_v2("v2_horizon"), "env", "ours (longer rent horizon)"),
    "v2_score":  (_b_v2("v2_score"), "env", "ours (deeds priced as the engine scores them)"),
    "v2_score_auc": (_b_v2("v2_score_auc"), "env", "ours (score pricing + auction ceiling)"),
    "v2_score_deep": (_b_v2("v2_score_deep"), "env", "ours (score pricing + thinner floor)"),
    "6c0de":     (_b_6c0de,    "env", "6c0de/exposure-monopoly-agent"),
    "expo":      (_b_expo,     "env", "emingurbuz9483/exposure-monopoly-algorithm"),
    "boom":      (_b_boom,     "sae", "EnzeCbe/monopoly-boom"),
    "inncenta":  (_b_inncenta, "env", "Inncenta/monopoly"),
    "aline":     (_b_aline,    "env", "alinebidal10-afk/monopoly-competition-agent"),
    "slayer":    (_b_slayer,   "env", "emirkaanozdemr/monopoly"),
}

# Ours are candidates for the submitted slot, never opposition.
OURS = ["UNDERDOG", "LAST_RESORT", "champion", "core", "stall", "thaw",
        "v2_plain", "v2_plain2", "v2_buy", "v2_plainbuy", "v2_horizon",
        "v2_score", "v2_score_auc", "v2_score_deep"]
RIVALS = [k for k in AGENTS if k not in OURS]


class Seat:
    """One agent bound to one seat, normalised to ``choose_action(env)``.

    Counts illegal and failed decisions instead of letting either end the game,
    and reports them: a team whose agent throws is a finding worth stating, not
    a crash to be absorbed silently.
    """

    def __init__(self, name: str, seat: int):
        builder, conv, _ = AGENTS[name]
        self.name = name
        self.seat = seat
        self.conv = conv
        self.agent = builder(seat)
        self.illegal = 0
        self.errors = 0

    def choose_action(self, env, legal: list[int]) -> int:
        from monopoly_game_engine.actions import ActionType
        try:
            if self.conv == "sae":
                state = env._get_state(self.seat)
                action = int(self.agent.choose_action(state, list(legal), env))
            else:
                action = int(self.agent.choose_action(env))
        except Exception:
            self.errors += 1
            action = -1
        if action not in legal:
            if action != -1:
                self.illegal += 1
            end = int(ActionType.END_TURN)
            action = end if end in legal else legal[0]
        return action


# ------------------------------------------------------------------ one game

_CACHE: dict[tuple, Seat] = {}


def _seat(name: str, seat: int) -> Seat:
    """Reuse agents inside a worker: EnzeCbe loads a 14 MB checkpoint.

    Every agent here reads its decision from ``env`` and the READMEs state no
    state is carried between games, so reuse changes no result -- it only
    avoids paying construction on every game.
    """
    key = (name, seat)
    if key not in _CACHE:
        _CACHE[key] = Seat(name, seat)
    obj = _CACHE[key]
    obj.illegal = 0
    obj.errors = 0
    return obj


def _play(lineup: tuple[str, ...], seed: int) -> dict:
    """Play one game. ``lineup[i]`` is the agent name occupying seat ``i``."""
    import random
    from monopoly_game_engine.env import MonopolyEnv

    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=MAX_ROUNDS)
    env.reset()
    seats = [_seat(n, i) for i, n in enumerate(lineup)]

    started = time.perf_counter()
    decisions = 0
    while not env.done and decisions < DECISION_CAP:
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        legal = env.get_allowed_actions(pid)
        if not legal:
            env._advance_turn()
            continue
        env.step(seats[pid].choose_action(env, legal))
        decisions += 1

    winner = env.winner()
    return {
        "lineup": list(lineup),
        "seed": seed,
        "winner": lineup[winner] if winner is not None else None,
        "winner_seat": winner,
        "truncated": env.round >= MAX_ROUNDS,
        "secs": time.perf_counter() - started,
        "illegal": {s.name: s.illegal for s in seats if s.illegal},
        "errors": {s.name: s.errors for s in seats if s.errors},
    }


def _play_baseline(name: str, seat: int, seed: int) -> dict:
    """One agent against three of the engine's fixed agents.

    A neutral third axis. Head-to-head numbers are relative, so a team that
    loses to us might still be strong or might be broken, and the two look
    identical from our win rate alone. Scoring everyone against the same weak
    reference separates those cases -- it is how we established that EnzeCbe's
    checkpoint is not merely weaker than ours but non-functional.
    """
    import random
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.env import MonopolyEnv
    from monopoly_game_engine.train import build_opponents

    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=MAX_ROUNDS)
    env.reset()
    agents = build_opponents(["fixed-a", "fixed-b", "fixed-c", "fixed-a"],
                             [0, 1, 2, 3])
    me = _seat(name, seat)
    agents[seat] = me

    started = time.perf_counter()
    decisions = 0
    while not env.done and decisions < DECISION_CAP:
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        legal = env.get_allowed_actions(pid)
        if not legal:
            env._advance_turn()
            continue
        if pid == seat:
            action = me.choose_action(env, legal)
        else:
            action = agents[pid].choose_action(env)
            if action not in legal:
                end = int(ActionType.END_TURN)
                action = end if end in legal else legal[0]
        env.step(action)
        decisions += 1

    winner = env.winner()
    return {"lineup": [name], "seed": seed,
            "winner": name if winner == seat else "fixed",
            "winner_seat": winner, "truncated": env.round >= MAX_ROUNDS,
            "secs": time.perf_counter() - started,
            "illegal": {me.name: me.illegal} if me.illegal else {},
            "errors": {me.name: me.errors} if me.errors else {}}


def _job(args):
    if args[0] == "__baseline__":
        _, name, seat, seed = args
        try:
            return _play_baseline(name, seat, seed)
        except Exception as exc:
            return {"lineup": [name], "seed": seed, "winner": None,
                    "winner_seat": None, "truncated": False, "secs": 0.0,
                    "illegal": {}, "errors": {},
                    "fatal": f"{type(exc).__name__}: {exc}"}
    lineup, seed = args
    try:
        return _play(tuple(lineup), seed)
    except Exception as exc:  # a broken agent must not kill the tournament
        return {"lineup": list(lineup), "seed": seed, "winner": None,
                "winner_seat": None, "truncated": False, "secs": 0.0,
                "illegal": {}, "errors": {}, "fatal": f"{type(exc).__name__}: {exc}"}


def _init():
    _pin_engine()


# ------------------------------------------------------------------ reporting

def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - m), 100 * (c + m))


def _build_jobs(mode: str, seeds: int, only: list[str] | None,
                exclude: list[str] | None = None,
                candidate: str = "UNDERDOG"):
    """Seat-balanced job list.

    Every seed is replayed with the lineup rotated through all four seats, so
    seat advantage cancels exactly rather than approximately.
    """
    rivals = [r for r in RIVALS if (not only or r in only)
              and r not in (exclude or ())]
    # Longest-job-first: measured cost per game spans 0.2 s to 68 s, because
    # 6c0de wraps ASU and inncenta uses ASU's value function. Queueing the slow
    # matchups first keeps them from becoming a tail that runs alone at the end
    # while nine workers idle.
    cost = {"6c0de": 68.0, "inncenta": 44.0, "expo": 3.1,
            "boom": 0.5, "aline": 0.2, "slayer": 0.2}
    rivals.sort(key=lambda r: -cost.get(r, 1.0))
    jobs = []
    if mode == "baseline":
        for name in ["UNDERDOG", *rivals]:
            for s in range(seeds):
                for seat in range(NUM_PLAYERS):
                    jobs.append(("__baseline__", name, seat, 9000 + s))
    elif mode == "h2h":
        # UNDERDOG against three copies of one rival: the direct question.
        for rival in rivals:
            for s in range(seeds):
                for seat in range(NUM_PLAYERS):
                    lineup = [rival] * NUM_PLAYERS
                    lineup[seat] = "UNDERDOG"
                    jobs.append((lineup, 5000 + s))
    elif mode == "melee":
        # Four distinct teams per table, the shape the real match uses.
        # Biased by construction -- see ``rr``, which is the one to trust.
        for combo in itertools.combinations(rivals, 3):
            table = [candidate, *combo]
            for s in range(seeds):
                for rot in range(NUM_PLAYERS):
                    jobs.append(([table[(i - rot) % 4] for i in range(4)],
                                 7000 + s))
    else:
        # Balanced round-robin: every 4-subset of the whole field, so each
        # agent plays the same number of tables and, more importantly, meets
        # the same distribution of opponents.
        #
        # ``melee`` puts UNDERDOG in every table, which quietly asks a
        # different question of each agent: UNDERDOG faces three rivals while
        # every rival faces UNDERDOG plus two rivals. If UNDERDOG is not
        # exactly average, those are different-strength fields and the
        # standings cannot be compared. Drawing tables from the full field
        # removes that asymmetry -- each of the n agents appears in
        # C(n-1, 3) tables and no agent is a fixed feature of the opposition.
        field = ["UNDERDOG", *rivals]
        for combo in itertools.combinations(field, NUM_PLAYERS):
            for s in range(seeds):
                for rot in range(NUM_PLAYERS):
                    jobs.append(([combo[(i - rot) % NUM_PLAYERS]
                                  for i in range(NUM_PLAYERS)], 7000 + s))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["h2h", "melee", "baseline", "rr"],
                    default="h2h")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--candidate", default="UNDERDOG",
                    help="which of ours fills the submitted slot in melee")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="drop agents from the field, e.g. a non-functional one")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    jobs = _build_jobs(a.mode, a.seeds, a.only, a.exclude, a.candidate)
    print(f"mode={a.mode}  {len(jobs)} games  workers={a.workers}  "
          f"parity={100/NUM_PLAYERS:.0f}%", flush=True)

    results = []
    started = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                mins = (time.perf_counter() - started) / 60
                print(f"  {i}/{len(jobs)}  {mins:.1f} min elapsed", flush=True)

    _report(a.mode, results)
    tag = a.mode if a.candidate == "UNDERDOG" else f"{a.mode}_{a.candidate}"
    out = Path(a.out) if a.out else (REPO / "artifacts" / f"gauntlet_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


def _report(mode: str, results: list[dict]) -> None:
    fatal = [r for r in results if r.get("fatal")]
    if fatal:
        print(f"\n{len(fatal)} games failed outright:")
        seen = set()
        for r in fatal:
            key = r["fatal"][:120]
            if key not in seen:
                seen.add(key)
                print(f"  {'/'.join(sorted(set(r['lineup'])))}: {key}")

    print(f"\n{'':<12}{'games':>7}{'wins':>7}{'win%':>8}{'95% CI':>16}"
          f"{'trunc%':>9}{'s/game':>9}")
    print("-" * 68)

    if mode == "h2h":
        for rival in RIVALS:
            rows = [r for r in results
                    if "UNDERDOG" in r["lineup"] and rival in r["lineup"]]
            if not rows:
                continue
            n = len(rows)
            w = sum(r["winner"] == "UNDERDOG" for r in rows)
            lo, hi = wilson(w, n)
            tr = 100 * sum(r["truncated"] for r in rows) / n
            sec = sum(r["secs"] for r in rows) / n
            print(f"vs {rival:<9}{n:>7}{w:>7}{100*w/n:>7.1f}%"
                  f"{f'[{lo:.1f}, {hi:.1f}]':>16}{tr:>8.0f}%{sec:>9.2f}")
    else:
        # Appearances, not games: every agent is not in every table.
        for name in [*OURS, *RIVALS]:
            rows = [r for r in results if name in r["lineup"]]
            if not rows:
                continue
            n = len(rows)
            w = sum(r["winner"] == name for r in rows)
            lo, hi = wilson(w, n)
            tr = 100 * sum(r["truncated"] for r in rows) / n
            sec = sum(r["secs"] for r in rows) / n
            print(f"{name:<12}{n:>7}{w:>7}{100*w/n:>7.1f}%"
                  f"{f'[{lo:.1f}, {hi:.1f}]':>16}{tr:>8.0f}%{sec:>9.2f}")

    bad_i, bad_e = {}, {}
    for r in results:
        for k, v in r.get("illegal", {}).items():
            bad_i[k] = bad_i.get(k, 0) + v
        for k, v in r.get("errors", {}).items():
            bad_e[k] = bad_e.get(k, 0) + v
    if bad_i or bad_e:
        print("\nillegal moves / raised exceptions (substituted, not fatal):")
        for k in sorted(set(bad_i) | set(bad_e)):
            print(f"  {k:<12} illegal={bad_i.get(k,0):<7} errors={bad_e.get(k,0)}")


if __name__ == "__main__":
    _pin_engine()
    main()
