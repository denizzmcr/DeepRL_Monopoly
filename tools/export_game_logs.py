"""
export_game_logs.py
-------------------
Exports complete game records: every action every player took, with dice,
phase, and the resulting standings.

The log is about the *game*, not about any agent's internals. All four seats are
named policies, so the same exporter records a match between any combination of
scripted agents, checkpoints, or a mix.

Two formats per game, from the same run:
  <name>.jsonl  one JSON object per event, for analysis
  <name>.txt    the same record as a readable transcript

Usage:
    python tools/export_game_logs.py --games 20 \
        --players ppo:artifacts/CHAMPION.pt fixed-a fixed-b fixed-c

    python tools/export_game_logs.py --games 5 --seed-base 500 \
        --players ppo:artifacts/CHAMPION.pt fixed-d fixed-e fixed-f \
        --out-dir artifacts/game_logs/champion_vs_def
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_game_engine.actions import ActionType, action_to_description  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS, RULESET_VERSION  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.train import build_opponents  # noqa: E402

MAX_STEPS = 200 * NUM_PLAYERS * 30


def standings(env):
    return [
        {
            "player": pid,
            "cash": int(env.players[pid].cash),
            "properties": len(env.players[pid].properties),
            "monopolies": int(env.players[pid].num_monopolies()),
            "net_worth": float(env.players[pid].net_worth()),
            "in_jail": bool(env.players[pid].in_jail),
            "bankrupt": bool(env.players[pid].bankrupt),
        }
        for pid in range(NUM_PLAYERS)
    ]


def play_and_record(policy_ids, seed, max_rounds=200):
    """Play one game, returning every event that occurred."""
    random.seed(seed)
    np.random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=max_rounds)
    env.reset()

    # Every seat is just a policy; build_opponents handles scripted names and
    # checkpoint paths identically, including each checkpoint's own settings.
    agents = {}
    for pid, spec in enumerate(policy_ids):
        agents[pid] = build_opponents([spec], [pid])[0]

    events = []
    for step in range(MAX_STEPS):
        if env.done:
            break
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue

        legal = list(env.get_allowed_actions(pid))
        if not legal:
            legal = [int(ActionType.DO_NOTHING)]
        before_cash = int(env.players[pid].cash)
        before_position = int(env.players[pid].position)
        dice_before = tuple(env.last_dice) if env.last_dice else None

        attempted = agents[pid].choose_action(env)
        action = attempted
        illegal = attempted not in legal
        if illegal:
            # Recorded rather than hidden: an illegal choice is a policy failure,
            # and the record keeps both what was attempted and what replaced it.
            action = (
                int(ActionType.END_TURN)
                if int(ActionType.END_TURN) in legal
                else legal[0]
            )

        env.step(action)
        dice_after = tuple(env.last_dice) if env.last_dice else None
        # last_dice persists between actions, so reporting it unconditionally
        # would show a roll on turns where no dice were thrown.
        rolled = list(dice_after) if dice_after != dice_before else None

        events.append(
            {
                "step": step,
                "round": int(env.round),
                "phase": str(env.phase),
                "player": pid,
                "action_id": int(action),
                "action": action_to_description(int(action)),
                "attempted_action_id": int(attempted) if illegal else None,
                "attempted_action": action_to_description(int(attempted)) if illegal else None,
                "legal_action_count": len(legal),
                "dice": rolled,
                "position_before": before_position,
                "position_after": int(env.players[pid].position),
                "cash_before": before_cash,
                "cash_after": int(env.players[pid].cash),
                "illegal_action_replaced": illegal,
            }
        )

    winner = env.winner()
    return {
        "seed": seed,
        "ruleset": RULESET_VERSION,
        "players": {str(i): policy_ids[i] for i in range(NUM_PLAYERS)},
        "rounds": int(env.round),
        "truncated_at_round_cap": bool(env.round >= max_rounds),
        "winner": int(winner) if winner is not None else None,
        "final_standings": standings(env),
        "events": events,
    }


def write_transcript(record, path):
    lines = [
        "=" * 70,
        f"Monopoly game record   ruleset={record['ruleset']}   seed={record['seed']}",
        f"exported {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 70,
        "Players:",
    ]
    for pid, spec in record["players"].items():
        lines.append(f"  seat {pid}: {spec}")
    lines.append("-" * 70)

    current_round = None
    for e in record["events"]:
        if e["round"] != current_round:
            current_round = e["round"]
            lines.append(f"\n--- ROUND {current_round} ---")
        dice = f" rolled {e['dice'][0]}+{e['dice'][1]}" if e["dice"] else ""
        moved = (
            f" [{e['position_before']}->{e['position_after']}]"
            if e["position_before"] != e["position_after"]
            else ""
        )
        cash = (
            f" cash {e['cash_before']}->{e['cash_after']}"
            if e["cash_before"] != e["cash_after"]
            else ""
        )
        flag = (
            f"  <replaced illegal choice: {e['attempted_action']}>"
            if e["illegal_action_replaced"] else ""
        )
        lines.append(
            f"  seat {e['player']}: {e['action']}{dice}{moved}{cash}"
            f"  ({e['legal_action_count']} legal){flag}"
        )

    lines.append("\n" + "-" * 70)
    lines.append(
        f"Winner: seat {record['winner']}"
        + ("  (200-round cap, decided on net worth)" if record["truncated_at_round_cap"] else "")
    )
    lines.append(f"Rounds: {record['rounds']}   Actions: {len(record['events'])}")
    for s in record["final_standings"]:
        lines.append(
            f"  seat {s['player']}: cash ${s['cash']}  properties {s['properties']}"
            f"  monopolies {s['monopolies']}  net worth ${s['net_worth']:.0f}"
            + ("  BANKRUPT" if s["bankrupt"] else "")
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export complete Monopoly game records")
    parser.add_argument(
        "--players", nargs=NUM_PLAYERS, required=True,
        help="four policies, e.g. ppo:artifacts/CHAMPION.pt fixed-a fixed-b fixed-c",
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "game_logs"))
    parser.add_argument("--rotate-seats", action="store_true",
                        help="rotate the first policy through all four seats")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []

    for i in range(args.games):
        seed = args.seed_base + i
        order = list(args.players)
        if args.rotate_seats:
            shift = i % NUM_PLAYERS
            order = order[-shift:] + order[:-shift] if shift else order
        record = play_and_record(order, seed, args.max_rounds)

        name = f"game_{seed:06d}"
        (out / f"{name}.jsonl").write_text(
            "\n".join(
                json.dumps(x) for x in
                [{"type": "header", **{k: v for k, v in record.items() if k != "events"}}]
                + [{"type": "event", **e} for e in record["events"]]
            ) + "\n"
        )
        write_transcript(record, out / f"{name}.txt")
        index.append({
            "game": name, "seed": seed, "winner": record["winner"],
            "winner_policy": record["players"][str(record["winner"])] if record["winner"] is not None else None,
            "rounds": record["rounds"], "actions": len(record["events"]),
            "truncated": record["truncated_at_round_cap"],
        })
        print(f"  {name}: winner seat {record['winner']} "
              f"({index[-1]['winner_policy']}), {record['rounds']} rounds, "
              f"{len(record['events'])} actions", flush=True)

    (out / "index.json").write_text(json.dumps({
        "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ruleset": RULESET_VERSION,
        "players": args.players,
        "games": index,
    }, indent=2))
    print(f"\n{len(index)} games written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
