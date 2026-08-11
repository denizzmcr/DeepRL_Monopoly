"""Tests for the packaged submission agent.

Three things are worth testing here and one of them cannot be tested in-process.

1. **ASU isolation.** ``submission_agent`` may not reach ``ASU_FROZEN_TEACHER``,
   directly or through anything it imports. This test module *does* import ASU
   (for the equivalence check below), so the isolation check runs in a fresh
   interpreter and inspects that interpreter's ``sys.modules``.
2. **Legality and latency.** Full seeded games, every move checked against the
   engine's own action list, with a p95 per-move latency report.
3. **Equivalence with the evaluator.** Every measured win rate we have came from
   ``ASU_FROZEN_TEACHER.evaluate``'s adapter. If the packaged agent picks a
   different action anywhere, those numbers do not describe what we ship. This
   asserts move-for-move agreement over full games.
"""

from __future__ import annotations

import copy
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_game_engine.action_filters import is_liquidation  # noqa: E402
from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.train import build_opponents  # noqa: E402
from submission_agent import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    CheckpointError,
    IllegalActionError,
    SubmissionAgent,
    checkpoint_sha256,
)

MAX_STEPS = 200 * NUM_PLAYERS * 30
TRAINED_TRIO = ("fixed-a", "fixed-b", "fixed-c")

# The per-move budget the agent is designed against. The match's real budget is
# unknown; this is the design target from CLAUDE.md section 1, used as a
# regression ceiling rather than as a claim about the competition.
LATENCY_BUDGET_SECONDS = 1.0

# Isolation is checked on the whole research tree, not just ASU: the submission
# should pull in the engine and nothing else. monopoly_bench is listed because
# its MonopolyZero bootstrap trains on ASU decisions, which is the exact thing
# the rules forbid.
FORBIDDEN_ROOTS = (
    "ASU_FROZEN_TEACHER",
    "monopoly_bench",
    "RL_CFR_MONOPOLYMODIFIED",
    "SLM_HANDMADE_MONOPOLY",
)

_ISOLATION_PROBE = """
import sys
sys.path.insert(0, {root!r})
import submission_agent
submission_agent.SubmissionAgent(0, {checkpoint!r})
roots = sorted({{name.split('.')[0] for name in sys.modules}})
print('IMPORTED_ROOTS=' + ','.join(roots))
"""


def play_game(agent, opponent_ids=TRAINED_TRIO, seed=0, max_rounds=200):
    """Play one seeded game with ``agent`` in its own seat.

    Returns the finished env, this agent's per-move latencies, and how many
    times each seat produced an action the engine had not offered.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = MonopolyEnv(agent_ids=[agent.player_id], max_rounds=max_rounds)
    others = [pid for pid in range(NUM_PLAYERS) if pid != agent.player_id]
    seats = {opp.player_id: opp for opp in build_opponents(opponent_ids, others)}
    seats[agent.player_id] = agent

    latencies: list[float] = []
    illegal = [0] * NUM_PLAYERS
    liquidations = 0

    for _ in range(MAX_STEPS):
        if env.done:
            break
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        legal = list(env.get_allowed_actions(pid))
        if not legal:
            legal = [int(ActionType.DO_NOTHING)]

        if pid == agent.player_id:
            forced = env.debt_player == pid
            started = time.perf_counter()
            action = agent.choose_action(env)
            latencies.append(time.perf_counter() - started)
            if is_liquidation(action) and not forced:
                liquidations += 1
        else:
            action = seats[pid].choose_action(env)

        if action not in legal:
            # Only the scripted opponents ever land here; the agent under test
            # raises instead. Recorded per seat so the assertion can be precise.
            illegal[pid] += 1
            action = (
                int(ActionType.END_TURN)
                if int(ActionType.END_TURN) in legal
                else legal[0]
            )
        env.step(action)

    return env, latencies, illegal, liquidations


def percentile(values, fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


class SubmissionAgentIsolationTests(unittest.TestCase):
    """The submission must not be able to reach ASU, even transitively."""

    def test_fresh_interpreter_never_imports_the_research_trees(self) -> None:
        probe = _ISOLATION_PROBE.format(
            root=str(ROOT), checkpoint=str(DEFAULT_CHECKPOINT)
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"probe failed:\n{completed.stdout}\n{completed.stderr}",
        )
        line = next(
            (
                text
                for text in completed.stdout.splitlines()
                if text.startswith("IMPORTED_ROOTS=")
            ),
            None,
        )
        self.assertIsNotNone(line, msg=completed.stdout)
        roots = set(line.partition("=")[2].split(","))

        self.assertNotIn(
            "ASU_FROZEN_TEACHER",
            roots,
            msg=(
                "submission_agent reached ASU_FROZEN_TEACHER. ASU may be a "
                "training opponent and a benchmark only; no ASU code may run at "
                "match time."
            ),
        )
        self.assertEqual(
            sorted(root for root in FORBIDDEN_ROOTS if root in roots),
            [],
            msg="the submission should import the game engine and nothing else",
        )
        # Sanity check on the probe itself: if the import silently did nothing,
        # the assertions above would pass for the wrong reason.
        self.assertIn("monopoly_game_engine", roots)
        self.assertIn("submission_agent", roots)


class SubmissionAgentCheckpointTests(unittest.TestCase):
    def test_reads_restrict_liquidation_from_the_checkpoint(self) -> None:
        agent = SubmissionAgent(0)
        self.assertTrue(agent.hybrid)
        self.assertTrue(
            agent.restrict_liquidation,
            msg=(
                "CHAMPION.pt was trained with voluntary liquidation masked out; "
                "running it unrestricted scores 0%"
            ),
        )
        self.assertEqual(
            agent.describe()["sha256"], checkpoint_sha256(DEFAULT_CHECKPOINT)
        )

    def test_flag_comes_from_the_file_not_from_a_default(self) -> None:
        """A checkpoint without the flag must load as unrestricted.

        That is how such checkpoints were trained, and it is the only way to
        show the flag is read rather than assumed.
        """
        payload = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "no_flag.pt"
            stripped = copy.deepcopy(payload)
            stripped["training_config"].pop("restrict_liquidation")
            torch.save(stripped, legacy)
            self.assertFalse(SubmissionAgent(0, legacy).restrict_liquidation)

            explicit = Path(directory) / "flag_false.pt"
            off = copy.deepcopy(payload)
            off["training_config"]["restrict_liquidation"] = False
            torch.save(off, explicit)
            self.assertFalse(SubmissionAgent(0, explicit).restrict_liquidation)

    def test_refuses_a_checkpoint_from_another_ruleset(self) -> None:
        payload = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "wrong_ruleset.pt"
            mutated = copy.deepcopy(payload)
            mutated["ruleset"] = "some-other-ruleset"
            torch.save(mutated, wrong)
            with self.assertRaises(CheckpointError):
                SubmissionAgent(0, wrong)

        with self.assertRaises(CheckpointError):
            SubmissionAgent(0, ROOT / "artifacts" / "does_not_exist.pt")


class SubmissionAgentPlayTests(unittest.TestCase):
    def test_seeded_games_produce_no_illegal_actions(self) -> None:
        """Full games in three seats, every move checked against the engine."""
        latencies: list[float] = []
        voluntary_liquidations = 0
        for seat, seed in ((0, 20250811), (1, 7), (2, 99)):
            agent = SubmissionAgent(seat)
            env, seat_latencies, illegal, liquidations = play_game(agent, seed=seed)
            self.assertEqual(
                illegal[seat],
                0,
                msg=f"seat {seat} produced an illegal action on seed {seed}",
            )
            self.assertGreater(len(seat_latencies), 0)
            self.assertGreater(env.round, 0)
            self.assertEqual(agent.decisions, len(seat_latencies))
            latencies.extend(seat_latencies)
            voluntary_liquidations += liquidations

        self.assertEqual(
            voluntary_liquidations,
            0,
            msg=(
                "the agent liquidated without being in debt; rule 3 is not being "
                "applied, and this checkpoint scores 0% without it"
            ),
        )

        p50 = percentile(latencies, 0.50)
        p95 = percentile(latencies, 0.95)
        print(
            f"\nSubmissionAgent per-move latency over {len(latencies)} moves: "
            f"p50 {p50 * 1000:.2f} ms, p95 {p95 * 1000:.2f} ms, "
            f"max {max(latencies) * 1000:.2f} ms"
        )
        self.assertLess(p95, LATENCY_BUDGET_SECONDS)

    def test_matches_the_evaluator_move_for_move(self) -> None:
        """The packaged agent must reproduce the policy we measured.

        Imported here rather than at module scope so it is obvious this is the
        test's dependency, not the submission's; the isolation test above proves
        the submission itself never touches it.
        """
        from ASU_FROZEN_TEACHER.evaluate import AgentFactory, parse_agent_spec

        seat = 0
        agent = SubmissionAgent(seat)
        reference = AgentFactory().build(
            parse_agent_spec(f"ppo:{DEFAULT_CHECKPOINT}"), seat
        )

        random.seed(4242)
        np.random.seed(4242)
        env = MonopolyEnv(agent_ids=[seat], max_rounds=200)
        others = [pid for pid in range(NUM_PLAYERS) if pid != seat]
        seats = {opp.player_id: opp for opp in build_opponents(TRAINED_TRIO, others)}

        compared = 0
        for _ in range(MAX_STEPS):
            if env.done:
                break
            pid = env.whose_turn()
            if env.players[pid].bankrupt:
                env._advance_turn()
                continue
            legal = list(env.get_allowed_actions(pid))
            if not legal:
                legal = [int(ActionType.DO_NOTHING)]

            if pid == seat:
                action = agent.choose_action(env)
                self.assertEqual(
                    action,
                    reference.choose_action(env),
                    msg=f"diverged from the evaluator at round {env.round}",
                )
                compared += 1
            else:
                action = seats[pid].choose_action(env)
                if action not in legal:
                    action = (
                        int(ActionType.END_TURN)
                        if int(ActionType.END_TURN) in legal
                        else legal[0]
                    )
            env.step(action)

        self.assertGreater(compared, 50)


class SubmissionAgentFailClosedTests(unittest.TestCase):
    def test_an_illegal_choice_raises_instead_of_being_replaced(self) -> None:
        random.seed(11)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        agent = SubmissionAgent(0)

        illegal_action = ACTION_SPACE_SIZE - 1  # a top auction bid, out of phase
        self.assertNotIn(illegal_action, env.get_allowed_actions(0))
        agent._argmax = lambda *args, **kwargs: illegal_action

        with self.assertRaises(IllegalActionError) as raised:
            agent.choose_action(env)
        self.assertIn(str(illegal_action), str(raised.exception))

    def test_no_legal_actions_raises(self) -> None:
        random.seed(11)
        env = MonopolyEnv(agent_ids=[0], max_rounds=200)
        env.get_allowed_actions = lambda pid=None: []
        with self.assertRaises(IllegalActionError):
            SubmissionAgent(0).choose_action(env)


if __name__ == "__main__":
    unittest.main()
