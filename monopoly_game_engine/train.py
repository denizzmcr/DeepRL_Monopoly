"""
Training loop (Section VII).

Trains one learning agent (PPO or DDQN, standard or hybrid) against
three fixed-policy opponents. Logs win rates every log_every games.

Call-site fixes applied (companions to agent_ppo.py fixes)
──────────────────────────────────────────────────────────
Fix 1 – choose_action() now returns 4 values for PPO: (action, log_prob,
    value, nn_allowed).  log_prob is None when the hybrid fixed policy
    fired; in that case we skip buffer storage entirely.

Fix 2 – store() now receives nn_allowed so it can record the per-step
    action mask alongside the transition.

Fix 3 – update() now receives (last_next_state, last_done) so that
    mid-game rollout boundaries are bootstrapped correctly by the critic
    instead of defaulting to zero.

Fix 5 – bounded potential shaping replaces repeated absolute state rewards.
    Each neural transition receives gamma*Phi(next)-Phi(current), and terminal
    transitions use zero terminal potential before the explicit win/loss bonus.
    This prevents policies from earning reward merely by taking extra actions.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .actions import ActionType
from .agents_fixed import FP_AGENT_CLASSES, FixedPolicyAgent, FPAgentA, FPAgentB, FPAgentC
from .constants import NUM_PLAYERS
from .env import MonopolyEnv

POTENTIAL_REWARD_LIMIT = 2.0

OPPONENT_IDS = (
    *(f"fixed-{letter}" for letter in "abcdef"),
    "asu-value-v1",
    "asu-rollout-v1",
    "kuzey",          # teammate's heuristic: strongest and fastest opponent we have
    "kuzey-plus",
)
DEFAULT_OPPONENT_IDS = ("fixed-a", "fixed-b", "fixed-c")


def normalize_opponent_plan(opponents):
    """Accept either one opponent triple or a list of triples to rotate through.

    Training against a single triple overfits to those personalities: an agent
    trained only on Fixed-A/B/C scored 86.5% against them and 56.5% against the
    unseen Fixed-D/E/F. Rotating triples per game widens the exposure without
    changing anything for callers that pass a single triple.
    """
    if not opponents:
        raise ValueError("opponents must not be empty")
    if isinstance(opponents[0], str):
        return [tuple(opponents)]
    return [tuple(triple) for triple in opponents]


class FrozenPolicyOpponent:
    """A saved checkpoint playing as an opponent, weights frozen.

    Lets past snapshots and rival agents sit in the other seats, which is how
    you train against a population rather than a fixed cast. Necessary here
    because agents trained against one opponent set turn out to be specialists:
    our best agent scored 86.5% against the trio it trained on and 19.0% in a
    four-way that differed only in which fourth player was present.

    Inference mirrors the evaluator: hybrid interception, the checkpoint's own
    liquidation setting, and deterministic argmax.
    """

    _cache: dict = {}

    def __init__(self, checkpoint: str, player_id: int):
        from .agent_ppo import PPOAgent

        self.player_id = player_id
        self.checkpoint = checkpoint
        if checkpoint not in FrozenPolicyOpponent._cache:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            agent = PPOAgent(
                player_id=int(payload["player_id"]),
                hybrid=bool(payload["hybrid"]),
                hidden_dim=int(payload["hidden_dim"]),
                device="cpu",
            )
            agent.load(checkpoint)
            agent.actor.eval()
            FrozenPolicyOpponent._cache[checkpoint] = agent
        self.agent = FrozenPolicyOpponent._cache[checkpoint]

    def choose_action(self, env) -> int:
        from .action_filters import restrict_actions
        from .actions import ACTION_SPACE_SIZE
        from .agent_ppo import fixed_accept_trade_decision, fixed_buy_decision

        pid = self.player_id
        allowed = list(env.get_allowed_actions(pid))
        if self.agent.hybrid:
            if int(ActionType.BUY_PROPERTY) in allowed and fixed_buy_decision(env, pid):
                return int(ActionType.BUY_PROPERTY)
            if (
                int(ActionType.ACCEPT_TRADE) in allowed
                and env._incoming_trade(pid) is not None
            ):
                return (
                    int(ActionType.ACCEPT_TRADE)
                    if fixed_accept_trade_decision(env, pid)
                    else int(ActionType.DECLINE_TRADE)
                )
            fixed = {int(ActionType.BUY_PROPERTY), int(ActionType.ACCEPT_TRADE)}
            allowed = [a for a in allowed if a not in fixed]
        if getattr(self.agent, "restrict_liquidation", False):
            allowed = restrict_actions(env, pid, allowed)
        if not allowed:
            return int(ActionType.DO_NOTHING)
        state = torch.as_tensor(
            env._get_state(pid), dtype=torch.float32
        ).unsqueeze(0)
        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[0, allowed] = True
        with torch.inference_mode():
            scores = self.agent.actor(state, mask).squeeze(0)
        return int(scores.argmax().item())


class KuzeyHeuristicOpponent:
    """Kuzey's hand-written heuristic, played as a league opponent.

    The strongest opponent available to us and nearly free: 77.5% against
    Fixed-A/B/C where our champion scores 55%, at 0.07 ms per decision against
    ASU's 57 ms. That combination is why the league can now afford strong
    opposition in most fields instead of rationing it.

    Imported lazily from external/, like ASU, so nothing in the submitted agent's
    import graph reaches it and so a missing external/ only breaks the leagues
    that ask for it.
    """

    _cache: dict = {}

    def __init__(self, player_id: int, variant: str = "champion"):
        import sys as _sys
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[1] / "external" / "kuzey" / "Kuzeys_heuristic"
        if not root.exists():
            raise ValueError(f"Kuzey heuristic not found at {root}")
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        # our engine is already imported by the time this runs, and their binder
        # is idempotent, so it will not replace it with their vendored copy
        import heuristic as _h

        self.player_id = player_id
        cls = {"champion": _h.Champion, "plus": _h.ChampionPlus,
               "spine": _h.Spine}.get(variant, _h.Champion)
        key = (variant, player_id)
        if key not in KuzeyHeuristicOpponent._cache:
            KuzeyHeuristicOpponent._cache[key] = cls()
        self.agent = KuzeyHeuristicOpponent._cache[key]

    def choose_action(self, env) -> int:
        return int(self.agent.choose_action(env, self.player_id, 0))


def build_opponents(ids, player_ids):
    """Instantiate the named opponent policies on the given seats.

    The vocabulary matches ``ASU_FROZEN_TEACHER.evaluate --opponents`` and
    ``monopoly_bench``, so a name means the same policy everywhere. Every
    returned agent exposes ``choose_action(env)`` and ``player_id``, which is
    all ``run_episode`` needs.

    ASU is imported lazily on purpose. ``monopoly_game_engine/__init__`` imports
    this module, so a module-level ASU import would drag the teacher into every
    process that touches the engine — including the submitted agent, which is
    required to be ASU-free.
    """
    ids = tuple(ids)
    if len(ids) != len(player_ids):
        raise ValueError(f"Need {len(player_ids)} opponents, got {len(ids)}")
    agents = []
    for identifier, pid in zip(ids, player_ids):
        if identifier.startswith("ppo:"):
            agents.append(FrozenPolicyOpponent(identifier[4:], pid))
            continue
        if identifier not in OPPONENT_IDS:
            raise ValueError(
                f"Unknown opponent {identifier!r}; expected one of {OPPONENT_IDS} "
                "or ppo:/path/to/checkpoint.pt"
            )
        if identifier.startswith("kuzey"):
            variant = "plus" if identifier.endswith("-plus") else "champion"
            agents.append(KuzeyHeuristicOpponent(pid, variant))
            continue
        if identifier.startswith("asu-"):
            from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1

            policy = ASURolloutV1 if identifier == "asu-rollout-v1" else ASUValueV1
            agents.append(policy(pid))
        else:
            agents.append(FP_AGENT_CLASSES[ord(identifier[-1]) - ord("a")](pid))
    return agents


def run_episode(
    env: MonopolyEnv,
    learning_agent,
    fp_agents: List[FixedPolicyAgent],
    agent_pid: int,
    is_ppo: bool,
    update_online: bool = True,
    reward_mode: str = "potential",
) -> Dict:
    """
    Run one complete game. The learning agent occupies position agent_pid,
    fixed-policy agents fill the other three slots.

    Returns metrics dict including:
      won, reward, steps, stats,
      trades_initiated, trades_accepted, trades_declined, properties_acquired
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    update_stats = {}

    # ── Per-episode metric counters ──────────────────────────────────────────
    trades_initiated = 0
    trades_accepted = 0
    trades_declined = 0
    properties_acquired = 0

    prev_prop_count = len(env.players[agent_pid].properties)

    agents_map = {fp.player_id: fp for fp in fp_agents}
    agents_map[agent_pid] = learning_agent

    pending_transition = None

    def potential_delta(start: float, terminal: bool = False) -> float:
        next_potential = 0.0 if terminal else env._compute_reward(agent_pid)
        if reward_mode == "absolute":
            # Bonjour et al. Eq. 1: the in-game reward is the relative net-worth
            # position itself at every step, and the terminal step pays only the
            # win/loss constant c (added separately by add_win_loss).
            #
            # This differs from the default in a way that may set our ceiling.
            # Potential-difference shaping telescopes: summed over an episode the
            # intermediate terms cancel and little gradient remains once the
            # coarse lessons are learned, which matches the flat learning curves
            # measured across 41 checkpoints. An absolute reward pays for being
            # ahead on every step, so the signal never washes out.
            return 0.0 if terminal else float(next_potential)
        gamma = getattr(learning_agent, "gamma", 0.99)
        decision_penalty = getattr(learning_agent, "decision_penalty", 0.0)
        return float(
            np.clip(
                gamma * next_potential - start - decision_penalty,
                -POTENTIAL_REWARD_LIMIT,
                POTENTIAL_REWARD_LIMIT,
            )
        )

    max_steps = env.max_rounds * NUM_PLAYERS * 30
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1
        pid = env.whose_turn()

        if env.players[pid].bankrupt:
            env._advance_turn()
            continue

        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [int(ActionType.DO_NOTHING)]

        if pid == agent_pid:
            # ── Learning agent ────────────────────────────────────────────
            if is_ppo:
                action, log_prob, value, nn_allowed = learning_agent.choose_action(
                    state, env, allowed
                )

                # A neural transition spans opponent and hybrid-policy actions
                # until the next state where the actor is actually consulted.
                if log_prob is not None and pending_transition is not None:
                    reward = potential_delta(pending_transition[4])
                    total_reward += reward
                    if update_online:
                        learning_agent.store(
                            pending_transition[0],
                            pending_transition[1],
                            pending_transition[2],
                            reward,
                            pending_transition[3],
                            False,
                            pending_transition[5],
                        )
                        if len(learning_agent.buffer) >= learning_agent.n_steps:
                            update_stats = learning_agent.update(
                                last_next_state=state,
                                last_done=False,
                            )
                            # The sampled action used the pre-update actor.
                            # Resample so the next rollout begins on-policy.
                            action, log_prob, value, nn_allowed = (
                                learning_agent.choose_action(state, env, allowed)
                            )
                    pending_transition = None
            else:
                action, nn_allowed = learning_agent.choose_training_action(
                    state, env, allowed
                )
                if nn_allowed is not None and pending_transition is not None:
                    reward = potential_delta(pending_transition[2])
                    total_reward += reward
                    if update_online:
                        learning_agent.store_transition(
                            pending_transition[0],
                            pending_transition[1],
                            reward,
                            state,
                            False,
                            nn_allowed,
                        )
                        update_stats = learning_agent.update()
                    pending_transition = None
                log_prob, value = 0.0, 0.0

            if action not in allowed:
                raise ValueError(
                    f"Learning agent selected illegal action {action}; allowed={allowed}"
                )

            # ── Count the action BEFORE stepping ──────────────────────────
            a = action
            buy_offset = int(ActionType.BUY_PROPERTY)
            acc_offset = int(ActionType.ACCEPT_TRADE)
            dec_offset = int(ActionType.DECLINE_TRADE)
            from .actions import OFFSETS as _OFF

            is_trade_offer = (
                _OFF["buy_trade"] <= a < _OFF["buy_trade"] + 252
                or _OFF["sell_trade"] <= a < _OFF["sell_trade"] + 252
                or _OFF["exch_trade"] <= a < _OFF["exch_trade"] + 2268
            )

            if a == buy_offset:
                properties_acquired += 1
            elif a == acc_offset:
                trades_accepted += 1
            elif a == dec_offset:
                trades_declined += 1
            elif is_trade_offer:
                trades_initiated += 1

            transition_state = state.copy()
            potential_before = env._compute_reward(agent_pid)
            next_state, _, done, info = env.step(action)

            # Detect property gained via accepted trade
            new_prop_count = len(env.players[agent_pid].properties)
            if a == acc_offset and new_prop_count > prev_prop_count:
                properties_acquired += new_prop_count - prev_prop_count
            prev_prop_count = new_prop_count

            steps += 1

            if is_ppo and log_prob is not None:
                pending_transition = (
                    transition_state,
                    action,
                    log_prob,
                    value,
                    potential_before,
                    nn_allowed,
                )
            elif not is_ppo and nn_allowed is not None:
                pending_transition = (
                    transition_state,
                    action,
                    potential_before,
                )
            state = next_state

        else:
            # ── Fixed-policy agent ────────────────────────────────────────
            agent = agents_map.get(pid)
            action = agent.choose_action(env) if agent else int(ActionType.END_TURN)
            if action not in allowed:
                action = (
                    int(ActionType.END_TURN)
                    if int(ActionType.END_TURN) in allowed
                    else allowed[0]
                )

            next_state, _, done, _ = env.step(action)
            state = next_state

    # ── Game over ─────────────────────────────────────────────────────────────
    winner = env.winner()
    won = winner == agent_pid

    if pending_transition is not None:
        reward = potential_delta(
            pending_transition[4] if is_ppo else pending_transition[2],
            terminal=True,
        )
        total_reward += reward
        if update_online and is_ppo:
            learning_agent.store(
                pending_transition[0],
                pending_transition[1],
                pending_transition[2],
                reward,
                pending_transition[3],
                True,
                pending_transition[5],
            )
        elif update_online:
            reward += getattr(learning_agent, "win_loss_bonus", 0.0) * (
                1 if won else -1
            )
            total_reward += getattr(learning_agent, "win_loss_bonus", 0.0) * (
                1 if won else -1
            )
            learning_agent.store_transition(
                pending_transition[0],
                pending_transition[1],
                reward,
                state,
                True,
                (),
            )

    if update_online:
        if is_ppo:
            learning_agent.add_win_loss(won)
            total_reward += getattr(learning_agent, "win_loss_bonus", 0.0) * (
                1 if won else -1
            )
            if len(learning_agent.buffer) > 0:
                update_stats.update(
                    learning_agent.update(last_next_state=state, last_done=True)
                )
        else:
            update_stats = learning_agent.update()
            learning_agent.finish_episode()

    return {
        "won": won,
        "reward": total_reward,
        "steps": steps,
        "stats": update_stats,
        "trades_initiated": trades_initiated,
        "trades_accepted": trades_accepted,
        "trades_declined": trades_declined,
        "properties_acquired": properties_acquired,
    }


def train(
    learning_agent,
    is_ppo: bool,
    hybrid: bool,
    n_games: int = 2000,
    log_every: int = 50,
    seed: int = 42,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    watchdog=None,
    opponents=DEFAULT_OPPONENT_IDS,
    rotate_seats: bool = False,
    keep_snapshots: bool = False,
    reward_mode: str = "potential",
) -> Dict:
    """
    Main training function.

    ``rotate_seats`` moves the learner through all four seats, one per game,
    which is what Bonjour et al. do ("randomize the turn order to remove any
    advantage one may get due to the player's position"). It matters here
    because the observation's deed-ownership slots use physical player IDs and
    do not rotate with the actor, so a learner pinned to one seat can key on
    seat identity. Evaluation is seat-balanced, so training in seat 0 only
    leaves a gap between training and evaluation conditions.

    ``opponents`` names the three policies the learner trains against. It
    defaults to Fixed-A/B/C, which is what this function always used. Naming
    ASU seats here is the supported way to train against a strong opponent —
    note it costs roughly two orders of magnitude in wall time per game.

    Returns:
        history: dict with win_rates (list per log_every games) and other metrics
    """
    random.seed(seed)
    np.random.seed(seed)

    agent_pid = learning_agent.player_id
    env = MonopolyEnv(agent_ids=[agent_pid], max_rounds=200)

    other_pids = [i for i in range(NUM_PLAYERS) if i != agent_pid]
    opponent_plan = normalize_opponent_plan(opponents)
    # Rotated by game index rather than sampled, so each triple gets an equal
    # share and a resumed run stays reproducible.
    fp_agents = build_opponents(opponent_plan[0], other_pids)

    history = defaultdict(list)
    wins_window = 0
    window_games = 0

    window_trades_initiated = 0
    window_trades_accepted = 0
    window_trades_declined = 0
    window_props_acquired = 0

    print(f"\n{'=' * 60}")
    print(
        f"Training {'Hybrid' if hybrid else 'Standard'} "
        f"{'PPO' if is_ppo else 'DDQN'} agent (player {agent_pid})"
    )
    print(f"Total games: {n_games}  |  Log every: {log_every}")
    print(f"{'=' * 60}")

    starting_games = int(getattr(learning_agent, "games_trained", 0))
    games_completed = 0
    for game_num in range(1, n_games + 1):
        absolute_game = starting_games + game_num
        episode_seed = seed + absolute_game - 1
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if watchdog is not None:
            try:
                watchdog.check()
            except RuntimeError as exc:
                if checkpoint_path:
                    path = Path(checkpoint_path)
                    emergency = path.with_name(
                        f"{path.stem}_emergency{path.suffix}"
                    )
                    learning_agent.save(str(emergency))
                    print(f"Memory watchdog stopped training: {exc}")
                    print(f"Emergency checkpoint: {emergency}")
                history["stopped_early"] = True
                history["stop_reason"] = str(exc)
                break

        game_pid = agent_pid
        if rotate_seats:
            game_pid = (agent_pid + absolute_game - 1) % NUM_PLAYERS
            # env returns state/reward for agent_ids[0], so it has to follow
            learning_agent.player_id = game_pid
            env.agent_ids = [game_pid]
            other_pids = [i for i in range(NUM_PLAYERS) if i != game_pid]

        if rotate_seats or len(opponent_plan) > 1:
            fp_agents = build_opponents(
                opponent_plan[(absolute_game - 1) % len(opponent_plan)], other_pids
            )

        try:
            result = run_episode(
                env, learning_agent, fp_agents, game_pid, is_ppo,
                reward_mode=reward_mode,
            )
        finally:
            # Restore before any checkpoint save: load() checks player_id
            # against the constructed agent, so a checkpoint written while the
            # learner sat in seat 2 would refuse to load into seat 0.
            learning_agent.player_id = agent_pid
        games_completed = game_num
        learning_agent.games_trained = absolute_game

        if (
            checkpoint_path
            and checkpoint_every > 0
            and absolute_game % checkpoint_every == 0
        ):
            learning_agent.save(checkpoint_path)
            if keep_snapshots:
                # Every checkpoint overwrites the same file, so a run whose win
                # rate peaks mid-way and then declines leaves no way back to the
                # good weights. That already cost us the peak of one 4000-game
                # run. Snapshots are opt-in because they cost ~14 MB each.
                path = Path(checkpoint_path)
                snapshot = path.with_name(
                    f"{path.stem}_g{absolute_game:06d}{path.suffix}"
                )
                learning_agent.save(str(snapshot))

        if result["won"]:
            wins_window += 1
        window_games += 1

        window_trades_initiated += result["trades_initiated"]
        window_trades_accepted += result["trades_accepted"]
        window_trades_declined += result["trades_declined"]
        window_props_acquired += result["properties_acquired"]

        if game_num % log_every == 0:
            win_rate = wins_window / window_games * 100

            avg_trades_init = window_trades_initiated / window_games
            avg_trades_acc = window_trades_accepted / window_games
            avg_trades_dec = window_trades_declined / window_games
            avg_props = window_props_acquired / window_games

            history["win_rates"].append(win_rate)
            history["games"].append(absolute_game)
            history["rewards"].append(result["reward"])
            history["avg_trades_initiated"].append(avg_trades_init)
            history["avg_trades_accepted"].append(avg_trades_acc)
            history["avg_trades_declined"].append(avg_trades_dec)
            history["avg_properties_acquired"].append(avg_props)

            eps_str = (
                f"  ε={learning_agent.epsilon:.3f}"
                if hasattr(learning_agent, "epsilon")
                else ""
            )
            print(
                f"  Game {absolute_game:5d} | "
                f"Win%: {win_rate:5.1f}%{eps_str} | "
                f"Props: {avg_props:.1f} | "
                f"Trades init/acc/dec: "
                f"{avg_trades_init:.1f}/{avg_trades_acc:.1f}/{avg_trades_dec:.1f}"
            )

            wins_window = 0
            window_games = 0
            window_trades_initiated = 0
            window_trades_accepted = 0
            window_trades_declined = 0
            window_props_acquired = 0

    history["resumed_from_games"] = starting_games
    history["games_completed_this_run"] = games_completed
    history["games_completed"] = int(
        getattr(learning_agent, "games_trained", games_completed)
    )
    return dict(history)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    learning_agent,
    is_ppo: bool,
    n_games: int = 2000,
    n_runs: int = 5,
    seed: int = 0,
    opponents=DEFAULT_OPPONENT_IDS,
    reward_mode: str = "potential",
) -> Dict:
    """
    Evaluate a trained agent over n_runs × n_games against ``opponents``.
    Sets epsilon=0 for DDQN automatically.
    Returns win rates plus per-game averages of all tracked metrics.
    """
    if hasattr(learning_agent, "epsilon"):
        learning_agent.epsilon = 0.0

    agent_pid = learning_agent.player_id
    env = MonopolyEnv(agent_ids=[agent_pid], max_rounds=200)
    other_pids = [i for i in range(NUM_PLAYERS) if i != agent_pid]
    fp_agents = build_opponents(opponents, other_pids)

    all_wins = []
    all_ti, all_ta, all_td, all_pa = [], [], [], []

    for run in range(n_runs):
        random.seed(seed + run)
        np.random.seed(seed + run)
        torch.manual_seed(seed + run)
        wins = 0
        run_ti = run_ta = run_td = run_pa = 0
        for _ in range(n_games):
            result = run_episode(
                env, learning_agent, fp_agents, agent_pid, is_ppo,
                update_online=False, reward_mode=reward_mode,
            )
            if result["won"]:
                wins += 1
            run_ti += result["trades_initiated"]
            run_ta += result["trades_accepted"]
            run_td += result["trades_declined"]
            run_pa += result["properties_acquired"]

        rate = wins / n_games * 100
        all_wins.append(rate)
        all_ti.append(run_ti / n_games)
        all_ta.append(run_ta / n_games)
        all_td.append(run_td / n_games)
        all_pa.append(run_pa / n_games)
        print(
            f"  Run {run + 1}/{n_runs}: "
            f"win={rate:.1f}%  "
            f"props={run_pa / n_games:.1f}  "
            f"trades init/acc/dec="
            f"{run_ti / n_games:.1f}/{run_ta / n_games:.1f}/{run_td / n_games:.1f}"
        )

    mean_wr = float(np.mean(all_wins))
    std_wr = float(np.std(all_wins))
    print(f"\n  Overall win rate: {mean_wr:.2f}% ± {std_wr:.2f}%")
    print(f"  Avg props/game : {np.mean(all_pa):.2f} ± {np.std(all_pa):.2f}")
    print(f"  Avg trades initiated/game: {np.mean(all_ti):.2f}")
    print(f"  Avg trades accepted/game : {np.mean(all_ta):.2f}")
    print(f"  Avg trades declined/game : {np.mean(all_td):.2f}")

    return {
        "win_rates": all_wins,
        "mean_win_rate": mean_wr,
        "std_win_rate": std_wr,
        "avg_properties_acquired": float(np.mean(all_pa)),
        "avg_trades_initiated": float(np.mean(all_ti)),
        "avg_trades_accepted": float(np.mean(all_ta)),
        "avg_trades_declined": float(np.mean(all_td)),
    }
