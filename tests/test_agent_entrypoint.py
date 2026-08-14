"""Packaging tests for ``agent.py``, the file the match harness loads.

These check the contract rather than the play. Strength is measured by
``tools/gauntlet.py`` against the other teams' agents and recorded in
``docs/GAUNTLET.md``; what is asserted here is that the submitted entry point
declares the right signature, returns only legal actions, leaves the global RNG
alone, stays inside its latency budget, imports none of the research trees, and
is actually serving the gradient-boosted policy rather than the fallback.
"""
from __future__ import annotations

import random
import re
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent as entrypoint  # noqa: E402

from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.train import build_opponents  # noqa: E402

TRIO = ["fixed-a", "fixed-b", "fixed-c", "fixed-a"]

# The submission must reach the engine and nothing else. ``monopoly_bench`` is
# on the list because its bootstrap trains on ASU decisions, which is the thing
# the competition rules forbid; the others are research trees that have no
# business in a match process.
FORBIDDEN_ROOTS = (
    "ASU_FROZEN_TEACHER",
    "monopoly_bench",
    "RL_CFR_MONOPOLYMODIFIED",
    "SLM_HANDMADE_MONOPOLY",
)

_PROBE = """
import sys
sys.path.insert(0, {root!r})
import agent
agent._policy()
print('FALLBACK=' + str(agent._FALLBACK))
print('ROOTS=' + ','.join(sorted({{n.split('.')[0] for n in sys.modules}})))
"""


def to_harness_state(env, pid: int, legal) -> dict:
    """Serialise the engine into exactly what the harness sends over the pipe.

    The tournament hands over plain JSON; nothing else is available at match
    time. Building the payload here rather than passing ``env`` is the whole
    point of these tests: the previous revision passed a live engine in its own
    suite, so every test was green while the shipped agent never once saw a
    board.
    """
    import json as _json
    sys.path.insert(0, str(ROOT / "underdog_gbm"))
    import engine_shim  # noqa: F401  binds ``engine`` before obs_lib needs it
    from obs_lib import build_obs
    payload = {
        "schema_version": "exposure-agent-v1",
        "ruleset_version": "ppo-plus-v2",
        "vector": [float(x) for x in build_obs(env, pid)],
        "decision_seed": 12345678901234,
        "actions": {str(int(a)): f"a{int(a)}" for a in legal},
        "board": {
            "round": int(env.round), "phase": str(env.phase),
            "current_turn": int(pid), "active_player": int(env.active_player_id()),
            "has_rolled": bool(env.has_rolled),
            "last_dice": list(env.last_dice), "done": bool(env.done),
            "houses_available": int(env.houses_available),
            "hotels_available": int(env.hotels_available),
            "auction": (None if env.auction_property_id is None else {
                "property_id": int(env.auction_property_id),
                "bidders": [int(b) for b in env.auction_bidders],
                "current_bidder": (None if env.auction_current_pid is None
                                   else int(env.auction_current_pid)),
                "high_bid": int(env.auction_high_bid),
                "high_bidder": (None if env.auction_high_bidder is None
                                else int(env.auction_high_bidder))}),
            "players": [
                {"player_id": i, "cash": float(env.players[i].cash),
                 "position": int(env.players[i].position),
                 "in_jail": bool(env.players[i].in_jail),
                 "jail_turns": int(env.players[i].jail_turns),
                 "gooj_card": bool(env.players[i].gooj_card),
                 "bankrupt": bool(env.players[i].bankrupt),
                 "net_worth": float(env.players[i].net_worth()),
                 "properties": [int(pr.square_id) for pr in env.players[i].properties]}
                for i in range(4)],
            "properties": [
                {"square_id": int(sq),
                 "owner": None if pr.owner is None else int(pr.owner),
                 "mortgaged": bool(pr.mortgaged), "houses": int(pr.houses),
                 "is_monopoly": bool(pr.is_monopoly)}
                for sq, pr in env.properties.items()],
        },
    }
    return _json.loads(_json.dumps(payload))   # must survive the wire


def play(seat: int, seed: int, max_rounds: int = 200):
    """One seeded game with the entry point in ``seat``.

    Returns the finished env, per-decision latencies, and the number of returns
    the engine had not offered.
    """
    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=max_rounds)
    env.reset()
    opponents = build_opponents(TRIO, [0, 1, 2, 3])

    latencies: list[float] = []
    illegal = 0
    decisions = 0
    while not env.done and decisions < 24000:
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        legal = env.get_allowed_actions(pid)
        if not legal:
            env._advance_turn()
            continue
        if pid == seat:
            started = time.perf_counter()
            state = to_harness_state(env, pid, list(legal))
            action = entrypoint.choose_action(state, pid, list(legal))
            latencies.append(time.perf_counter() - started)
            if action not in legal:
                illegal += 1
        else:
            action = opponents[pid].choose_action(env)
            if action not in legal:
                action = legal[0]
        env.step(action)
        decisions += 1
    return env, latencies, illegal


class TestContract(unittest.TestCase):
    def test_declares_the_harness_signature_exactly(self) -> None:
        """``(state, player_id, allowed_actions)``, in that order, no extras.

        The previous revision declared ``(state, allowed_actions, env,
        player_id)`` and sorted the arguments by shape at runtime. That
        tolerance layer existed to survive a contract that was never
        ambiguous, and it hid the real defect for a whole tournament.
        """
        import inspect
        for target in (entrypoint.choose_action,
                       entrypoint.Agent(0).choose_action):
            params = list(inspect.signature(target).parameters)
            with self.subTest(target=target):
                self.assertEqual(params[:3],
                                 ["state", "player_id", "allowed_actions"])
                self.assertNotIn("env", params)

    def test_the_policy_runs_on_a_plain_dict(self) -> None:
        """The regression test for the defect that scored 0 wins in 1000 games.

        ``state`` is a dict. Detecting the board with ``hasattr`` returns False
        on every decision, and the agent then plays ``allowed_actions[0]``
        forever without anything noticing. ``STRICT`` turns that silent
        fallback into a failure.
        """
        random.seed(7)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        env.reset()
        pid = env.whose_turn()
        legal = list(env.get_allowed_actions(pid))
        state = to_harness_state(env, pid, legal)

        entrypoint.STRICT = True
        try:
            action = entrypoint.choose_action(state, pid, legal)
        finally:
            entrypoint.STRICT = False
        self.assertIn(action, legal)

    def test_the_ported_board_agrees_with_the_live_engine(self) -> None:
        """The dict rebuild must not change what the policy plays.

        Replays a game and asks the policy the same question twice: once from
        the live engine, once from the board rebuilt out of JSON. Any
        divergence means a feature is reading something the payload did not
        carry.
        """
        sys.path.insert(0, str(ROOT / "underdog_gbm"))
        import dict_board as dbmod

        policy = entrypoint._policy()
        random.seed(5)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        env.reset()
        compared = agreed = 0
        for _ in range(600):
            if env.done:
                break
            pid = env.whose_turn()
            if env.players[pid].bankrupt:
                env._advance_turn()
                continue
            legal = sorted(int(a) for a in env.get_allowed_actions(pid))
            if not legal:
                env._advance_turn()
                continue
            if len(legal) == 1:
                env.step(legal[0])
                continue
            live = policy.choose_action(env, pid, 0)
            state = to_harness_state(env, pid, legal)
            board = dbmod.board_from_state(state, legal, pid)
            self.assertIsNotNone(board, "board_from_state found no board")
            compared += 1
            agreed += int(policy.choose_action(board, pid, 0) == live)
            env.step(live)
        self.assertGreater(compared, 200, "too few decisions to be evidence")
        self.assertEqual(agreed, compared,
                         f"ported board diverged on {compared - agreed} of "
                         f"{compared} decisions")

    def test_never_redeems_a_mortgage_while_in_debt(self) -> None:
        """The exact livelock that bankrupted the previous revision in round 0.

        In debt the engine offers only liquidation actions and reorders them
        after each one, so ``allowed_actions[0]`` alternates mortgage /
        unmortgage and bleeds 10% per cycle. The guard refuses the redeem leg
        outright, so the cycle cannot form.
        """
        import json as _json
        base = {
            "schema_version": "exposure-agent-v1",
            "ruleset_version": "ppo-plus-v2",
            "vector": [0.0] * 300, "decision_seed": 7,
            "actions": {"11": "mortgage(sq=11)", "39": "unmortgage(sq=11)"},
            "board": {
                "round": 0, "phase": "post_roll", "current_turn": 0,
                "active_player": 0, "has_rolled": True, "last_dice": [3, 4],
                "done": False, "houses_available": 32, "hotels_available": 12,
                "auction": None,
                "players": [{"player_id": i, "cash": 0, "position": 11,
                             "in_jail": False, "jail_turns": 0,
                             "gooj_card": False, "bankrupt": False,
                             "net_worth": 60, "properties": []}
                            for i in range(4)],
                "properties": [{"square_id": 11, "owner": 0, "mortgaged": False,
                                "houses": 0, "is_monopoly": False}],
            },
        }
        seat = entrypoint.Agent(0)
        emitted, order = [], [11, 39]
        for i in range(60):
            state = _json.loads(_json.dumps(base))
            state["board"]["properties"][0]["mortgaged"] = bool(i % 2)
            emitted.append(seat.choose_action(state, 0, list(order)))
            order = order[::-1]          # the engine reorders after each one
        self.assertNotIn(39, emitted, "redeemed a mortgage while in debt")
        flips = sum(1 for i in range(1, len(emitted))
                    if emitted[i] != emitted[i - 1])
        self.assertLess(flips, 3, f"oscillated {flips} times: {emitted[:12]}")

    def test_a_missing_board_is_survived_not_hidden(self) -> None:
        """Malformed payloads must return a legal action, and say so once."""
        for bad in ({}, {"board": None}, {"board": {"players": []}}, "not a dict"):
            with self.subTest(payload=str(bad)[:24]):
                self.assertIn(entrypoint.choose_action(bad, 0, [7, 11, 13]),
                              (7, 11, 13))

    def test_exports_both_calling_conventions(self) -> None:
        self.assertTrue(callable(entrypoint.choose_action))
        self.assertTrue(callable(entrypoint.make_agent))
        self.assertTrue(hasattr(entrypoint.Agent(0), "choose_action"))

    def test_serves_the_gradient_boosted_policy_not_the_fallback(self) -> None:
        """The measured agent is the one that ships.

        The fallback exists so an environment failure costs strength instead of
        forfeiting every game, but if it is silently in use here then the win
        rates in docs/GAUNTLET.md describe a policy nobody is running.
        """
        policy = entrypoint._policy()
        self.assertFalse(entrypoint._FALLBACK,
                         "fell back to the rule agent; LightGBM or the boosters "
                         "failed to load")
        self.assertEqual(type(policy).__name__, "MonopolyAgent")


class TestLegality(unittest.TestCase):
    def test_seeded_games_from_every_seat_produce_no_illegal_actions(self) -> None:
        for seat in range(4):
            with self.subTest(seat=seat):
                env, latencies, illegal = play(seat, seed=4242 + seat)
                self.assertGreater(len(latencies), 100,
                                   "game ended too early to be evidence")
                self.assertEqual(illegal, 0)
                self.assertTrue(env.done)

    def test_an_empty_allowed_set_still_returns_an_int(self) -> None:
        self.assertIsInstance(entrypoint.choose_action({}, 0, []), int)

    def test_a_board_less_call_returns_a_legal_action(self) -> None:
        """A payload with no board must not raise."""
        self.assertIn(entrypoint.choose_action({}, 0, [7, 11, 13]), (7, 11, 13))


class TestDeterminismAndRng(unittest.TestCase):
    def test_the_global_rng_is_never_advanced(self) -> None:
        """A submission that draws from the shared stream desynchronises dice."""
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        random.seed(99)
        env.reset()
        pid = env.whose_turn()
        legal = list(env.get_allowed_actions(pid))

        random.seed(1234)
        np.random.seed(1234)
        before = (random.getstate(), np.random.get_state()[1][0])
        entrypoint.choose_action(to_harness_state(env, pid, legal), pid, legal)
        after = (random.getstate(), np.random.get_state()[1][0])
        self.assertEqual(before[0], after[0], "python's global RNG moved")
        self.assertEqual(before[1], after[1], "numpy's global RNG moved")

    def test_the_same_position_gives_the_same_action(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        random.seed(7)
        env.reset()
        pid = env.whose_turn()
        legal = list(env.get_allowed_actions(pid))
        state = to_harness_state(env, pid, legal)
        first = entrypoint.choose_action(state, pid, legal)
        for _ in range(5):
            self.assertEqual(entrypoint.choose_action(state, pid, legal), first)


class TestLatency(unittest.TestCase):
    BUDGET_S = 1.0

    def test_p95_stays_far_inside_the_budget(self) -> None:
        _, latencies, _ = play(0, seed=11)
        ms = sorted(x * 1000 for x in latencies)
        p50 = statistics.median(ms)
        p95 = ms[int(0.95 * (len(ms) - 1))]
        print(f"\n  latency over {len(ms)} decisions: "
              f"p50 {p50:.2f} ms  p95 {p95:.2f} ms  max {ms[-1]:.1f} ms")
        self.assertLess(p95 / 1000, self.BUDGET_S)


class TestIsolation(unittest.TestCase):
    def test_a_fresh_interpreter_imports_none_of_the_research_trees(self) -> None:
        """ASU may train and benchmark this agent; it may not be inside it."""
        probe = _PROBE.format(root=str(ROOT))
        completed = subprocess.run([sys.executable, "-c", probe],
                                   capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        line = next(ln for ln in completed.stdout.splitlines()
                    if ln.startswith("ROOTS="))
        roots = set(line.split("=", 1)[1].split(","))
        self.assertEqual(sorted(roots & set(FORBIDDEN_ROOTS)), [])

    def test_the_fresh_interpreter_also_loads_the_real_policy(self) -> None:
        """Isolation is worthless if it only holds for the fallback path."""
        probe = _PROBE.format(root=str(ROOT))
        completed = subprocess.run([sys.executable, "-c", probe],
                                   capture_output=True, text=True, cwd=str(ROOT))
        self.assertIn("FALLBACK=False", completed.stdout, completed.stderr)

    def test_the_agent_adds_no_heavy_dependency_of_its_own(self) -> None:
        """What we cost the sandbox is the delta over the engine, not the total.

        ``monopoly_game_engine/__init__.py`` imports torch itself, so torch is in
        any process that can hold an ``env`` at all -- including this one, before
        ``agent`` is imported. That is the harness's dependency and not ours, and
        it is why ``requirements.txt`` does not list it: the Linux ``torch`` wheel
        pulls the CUDA stack and would not fit the 2 GiB cap on its own.

        The number this test defends is the one we control: everything our agent
        adds on top of the engine.
        """
        # Our own modules load as top-level names by design -- see
        # underdog_gbm/__init__.py for why they are not renamed.
        OURS = {"agent", "underdog_gbm", "engine", "engine_shim", "act_lib",
                "v2_lib", "obs_lib", "feature_lib", "ing_features"}
        # What LightGBM brings with it. Together ~60 MB installed.
        # ``cython_runtime`` is not a distribution -- Cython injects it into
        # sys.modules when a compiled scipy extension loads.
        EXPECTED = {"lightgbm", "scipy", "narwhals", "cython_runtime"}

        baseline = self._roots("import monopoly_game_engine")
        withagent = self._roots("import monopoly_game_engine\nimport agent\n"
                                "agent._policy()")
        added = {r for r in withagent - baseline
                 if not r.startswith("_") and r not in sys.stdlib_module_names}

        self.assertIn("torch", baseline, "engine no longer imports torch; re-check "
                                         "whether requirements.txt should list it")
        self.assertEqual(sorted(added - OURS), sorted(EXPECTED),
                         "the agent pulled in a third-party package that "
                         "requirements.txt does not account for")

    def test_importing_us_does_not_put_a_second_engine_on_the_path(self) -> None:
        """``underdog/engine/`` is a full vendored copy of the simulator.

        If ``agent.py`` put ``underdog/`` on sys.path at import time, a later
        ``import engine`` anywhere in the harness process would find our copy
        and silently decide the rules for the whole table -- the failure this
        project pinned its own engine to avoid. Nothing we do to sys.path may
        shadow a name the harness owns.
        """
        script = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
                  "import agent, importlib.util\n"
                  "spec = importlib.util.find_spec('engine')\n"
                  "print('ENGINE=' + (spec.origin or '') if spec else 'ENGINE=')")
        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        origin = completed.stdout.split("ENGINE=", 1)[1].strip()
        self.assertNotIn("underdog/engine", origin,
                         "importing agent.py exposed the vendored engine copy")

    def test_the_fallback_binds_the_harness_engine_not_the_vendored_one(self) -> None:
        """The fallback is the path most likely to get this wrong, so pin it."""
        script = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
                  "import agent\n"
                  "p = agent._fallback()\n"
                  "print('ENGINE=' + sys.modules['engine'].__file__)")
        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        origin = completed.stdout.split("ENGINE=", 1)[1].strip()
        self.assertIn("monopoly_game_engine", origin)
        self.assertNotIn("underdog/engine", origin)

    def _roots(self, body: str) -> set[str]:
        script = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n{body}\n"
                  "print('ROOTS=' + ','.join(sorted({n.split('.')[0] "
                  "for n in sys.modules})))")
        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        line = next(ln for ln in completed.stdout.splitlines()
                    if ln.startswith("ROOTS="))
        return set(line.split("=", 1)[1].split(","))


class TestRequirements(unittest.TestCase):
    """The rules cap requirements.txt at 32 wheel-only PyPI entries."""

    MAX_ENTRIES = 32

    def entries(self) -> list[str]:
        text = (ROOT / "requirements.txt").read_text().splitlines()
        return [ln.strip() for ln in text
                if ln.strip() and not ln.strip().startswith("#")]

    def test_within_the_entry_cap(self) -> None:
        self.assertLessEqual(len(self.entries()), self.MAX_ENTRIES)

    def test_every_entry_is_a_plain_pinned_pypi_name(self) -> None:
        """No VCS URLs, no local paths, no --find-links: those are not wheels."""
        pattern = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?"
                             r"([<>=!~]=?[0-9A-Za-z.\-*]+)(,[<>=!~]=?[0-9A-Za-z.\-*]+)*$")
        for entry in self.entries():
            with self.subTest(entry=entry):
                self.assertRegex(entry, pattern)

    def test_lists_what_the_agent_actually_imports(self) -> None:
        names = {e.split("[")[0].split("<")[0].split(">")[0].split("=")[0].lower()
                 for e in self.entries()}
        self.assertIn("lightgbm", names)
        self.assertIn("numpy", names)


if __name__ == "__main__":
    unittest.main()
