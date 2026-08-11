"""Tests for PPO with parallel actors.

The properties worth pinning down:

* the wire format is lossless — a worker's rollout has to arrive at the learner
  byte-identical, or the update is computed on data nobody collected;
* workers really do not learn, so a round's games are all on the same policy;
* the whole run is reproducible despite being parallel, which is what makes a
  result comparable to the sequential trainer's;
* a dead or failing worker surfaces as an error instead of a hang.
"""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_game_engine.actions import ACTION_SPACE_SIZE  # noqa: E402
from monopoly_game_engine.agent_ppo import PPOAgent  # noqa: E402
from monopoly_game_engine.parallel_ppo import (  # noqa: E402
    _NEVER,
    _CollectingAgent,
    _pack_trajectory,
    _restore_trajectory,
    train_parallel,
)

TRAINED_TRIO = ("fixed-a", "fixed-b", "fixed-c")


def _fake_rollout(agent, steps: int, rng: random.Random) -> None:
    for step in range(steps):
        allowed = sorted(rng.sample(range(ACTION_SPACE_SIZE), rng.randint(1, 12)))
        agent.store(
            np.arange(300, dtype=np.float32) * (step + 1) / 300.0,
            allowed[0],
            -1.234 * (step + 1),
            0.5 * step,
            0.25 * step,
            step == steps - 1,
            allowed,
        )


class TrajectoryWireFormatTests(unittest.TestCase):
    def test_pack_and_restore_is_lossless(self) -> None:
        source = _CollectingAgent(
            player_id=0, hybrid=True, n_steps=_NEVER, device="cpu"
        )
        _fake_rollout(source, 40, random.Random(3))

        destination = PPOAgent(player_id=0, hybrid=True, device="cpu")
        restored = _restore_trajectory(destination, _pack_trajectory(source.buffer))

        self.assertEqual(restored, 40)
        self.assertEqual(len(destination.buffer), len(source.buffer))
        self.assertEqual(destination.buffer.actions, source.buffer.actions)
        self.assertEqual(destination.buffer.dones, source.buffer.dones)
        for original, copied in zip(source.buffer.states, destination.buffer.states):
            np.testing.assert_allclose(original, copied)
        for original, copied in zip(
            source.buffer.log_probs, destination.buffer.log_probs
        ):
            self.assertAlmostEqual(original, copied, places=5)
        for original, copied in zip(
            source.buffer.action_masks, destination.buffer.action_masks
        ):
            self.assertTrue(torch.equal(original, copied))

    def test_masks_survive_the_bit_packing(self) -> None:
        """ACTION_SPACE_SIZE is not a multiple of 8; the tail must not be lost."""
        self.assertNotEqual(ACTION_SPACE_SIZE % 8, 0)
        agent = _CollectingAgent(player_id=0, hybrid=True, n_steps=_NEVER, device="cpu")
        edges = [0, 1, ACTION_SPACE_SIZE - 2, ACTION_SPACE_SIZE - 1]
        agent.store(
            np.zeros(300, dtype=np.float32), edges[0], 0.0, 0.0, 0.0, True, edges
        )
        packed = _pack_trajectory(agent.buffer)
        unpacked = np.unpackbits(packed["masks"], axis=1, count=ACTION_SPACE_SIZE)
        self.assertEqual(np.flatnonzero(unpacked[0]).tolist(), edges)


class CollectingAgentTests(unittest.TestCase):
    def test_collecting_agent_never_learns_and_never_flushes(self) -> None:
        agent = _CollectingAgent(player_id=0, hybrid=True, n_steps=_NEVER, device="cpu")
        before = [p.detach().clone() for p in agent.actor.parameters()]
        _fake_rollout(agent, 8, random.Random(5))

        self.assertEqual(agent.update(), {})
        self.assertEqual(len(agent.buffer), 8)
        for original, current in zip(before, agent.actor.parameters()):
            self.assertTrue(torch.equal(original, current))
        # run_episode's mid-episode flush is gated on this comparison.
        self.assertLess(len(agent.buffer), agent.n_steps)


class ParallelRunTests(unittest.TestCase):
    """End-to-end runs. Kept tiny — spawning workers costs more than the games."""

    GAMES = 2
    WORKERS = 2
    MAX_ROUNDS = 20

    def _run(self, directory: Path, name: str) -> tuple[dict, Path]:
        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)
        agent = PPOAgent(
            player_id=0,
            hybrid=True,
            entropy_coef=0.005,
            restrict_liquidation=True,
            device="cpu",
        )
        checkpoint = directory / f"{name}.pt"
        history = train_parallel(
            agent,
            n_games=self.GAMES,
            workers=self.WORKERS,
            games_per_round=self.GAMES,
            opponents=TRAINED_TRIO,
            seed=77,
            max_rounds=self.MAX_ROUNDS,
            log_every=self.GAMES,
            checkpoint_path=str(checkpoint),
            verbose=False,
        )
        return history, checkpoint

    def test_two_identical_runs_agree_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first, first_path = self._run(directory, "first")
            second, second_path = self._run(directory, "second")

            self.assertEqual(first["games_completed"], self.GAMES)
            self.assertEqual(first["games_completed_this_run"], self.GAMES)
            self.assertGreater(first["transitions_collected"], 0)
            self.assertEqual(
                first["transitions_collected"], second["transitions_collected"]
            )
            self.assertEqual(first["win_rates"], second["win_rates"])

            left = torch.load(first_path, map_location="cpu", weights_only=True)
            right = torch.load(second_path, map_location="cpu", weights_only=True)
            self.assertEqual(left["step_count"], right["step_count"])
            self.assertEqual(left["games_trained"], self.GAMES)
            self.assertTrue(left["training_config"]["restrict_liquidation"])
            for key, tensor in left["actor"].items():
                self.assertTrue(
                    torch.equal(tensor, right["actor"][key]),
                    msg=f"actor.{key} differs between two identical runs",
                )

    def test_the_checkpoint_reloads_into_a_normal_agent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            _, path = self._run(Path(raw), "reload")
            agent = PPOAgent(player_id=0, hybrid=True, device="cpu")
            agent.load(str(path))
            self.assertEqual(agent.games_trained, self.GAMES)
            self.assertTrue(agent.restrict_liquidation)

    def test_a_failing_worker_raises_instead_of_hanging(self) -> None:
        agent = PPOAgent(player_id=0, hybrid=True, device="cpu")
        with self.assertRaises(RuntimeError) as raised:
            train_parallel(
                agent,
                n_games=1,
                workers=1,
                games_per_round=1,
                opponents=("not-a-real-agent", "fixed-a", "fixed-b"),
                max_rounds=self.MAX_ROUNDS,
                verbose=False,
            )
        self.assertIn("not-a-real-agent", str(raised.exception))

    def test_rejects_impossible_configurations(self) -> None:
        agent = PPOAgent(player_id=0, hybrid=True, device="cpu")
        for kwargs in (
            {"workers": 0},
            {"n_games": 0},
            {"games_per_round": -1},
            {"games_per_round": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    train_parallel(
                        agent,
                        **{"n_games": 1, "workers": 1, "verbose": False, **kwargs},
                    )


if __name__ == "__main__":
    unittest.main()
