"""
monopoly_game_engine – Shared ppo-plus-v2 Monopoly simulator
=============================================================

Based on:
  "Decision Making in Monopoly Using a Hybrid Deep Reinforcement
   Learning Approach"
  Bonjour et al., IEEE TETCI, Vol. 6, No. 6, December 2022.

Quick start
-----------
>>> from monopoly_game_engine import train_ppo, train_ddqn, evaluate_agent
>>> agent, history = train_ppo(hybrid=True, n_games=2000)
>>> results = evaluate_agent(agent, is_ppo=True, n_games=2000)
"""

import random

import numpy as np

from .env          import MonopolyEnv
from .agents_fixed import FPAgentA, FPAgentB, FPAgentC
from .state        import build_state_vector
from .actions      import ACTION_SPACE_SIZE, action_to_description

# The trainers are the only part of this package that needs torch, and a
# submitted agent never touches them: it plays through `env`, `state`,
# `actions` and `constants`, none of which import it.
#
# The agent container is stock slim Python plus wheels resolved from the
# submission's own requirements.txt -- numpy and lightgbm, no torch. Importing
# these unconditionally made `import monopoly_game_engine` raise there, which
# the shim reported as a missing `engine` module, which the validator tried to
# install from PyPI, which aborted the build before a single game ran.
#
# So they are optional. Present for training, absent in the match sandbox.
try:
    import torch
    from .agent_ppo import PPOAgent
    from .agent_ddqn import DDQNAgent
    from .train import DEFAULT_OPPONENT_IDS, OPPONENT_IDS, train, evaluate
    HAS_TORCH = True
except ImportError:  # match sandbox: no torch, and nothing here needs it
    torch = None
    PPOAgent = DDQNAgent = train = evaluate = None
    DEFAULT_OPPONENT_IDS = OPPONENT_IDS = ()
    HAS_TORCH = False


def train_ppo(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 2000,
    log_every: int = 100,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    seed: int = 42,
    resume_path: str | None = None,
    opponents=DEFAULT_OPPONENT_IDS,
    rotate_seats: bool = False,
    **kwargs,
):
    """Train a PPO agent. Set hybrid=True for the hybrid approach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = PPOAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    if resume_path is not None:
        agent.load(resume_path)
        n_games = max(0, n_games - agent.games_trained)
    history = train(
        agent,
        is_ppo=True,
        hybrid=hybrid,
        n_games=n_games,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
        watchdog=watchdog,
        seed=seed,
        opponents=opponents,
        rotate_seats=rotate_seats,
    )
    return agent, history


def train_ddqn(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 10_000,
    log_every: int = 100,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    seed: int = 42,
    resume_path: str | None = None,
    opponents=DEFAULT_OPPONENT_IDS,
    rotate_seats: bool = False,
    **kwargs,
):
    """Train a DDQN agent. Set hybrid=True for the hybrid approach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DDQNAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    if resume_path is not None:
        agent.load(resume_path)
        n_games = max(0, n_games - agent.games_trained)
    history = train(
        agent,
        is_ppo=False,
        hybrid=hybrid,
        n_games=n_games,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
        watchdog=watchdog,
        seed=seed,
        opponents=opponents,
        rotate_seats=rotate_seats,
    )
    return agent, history


def evaluate_agent(agent, is_ppo: bool, n_games: int = 2000, n_runs: int = 5):
    """Evaluate a trained agent against fixed-policy opponents."""
    return evaluate(agent, is_ppo=is_ppo, n_games=n_games, n_runs=n_runs)


__all__ = [
    "MonopolyEnv",
    "PPOAgent", "DDQNAgent",
    "FPAgentA", "FPAgentB", "FPAgentC",
    "train_ppo", "train_ddqn", "evaluate_agent",
    "build_state_vector", "ACTION_SPACE_SIZE", "action_to_description",
]
