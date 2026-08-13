"""Distil Kuzey's heuristic into a network: collect labelled decisions, then fit.

Why this policy and not ASU: ASU is the instructor's and imitating it is
forbidden. Kuzey's heuristic is the team's own work, and the instructor approved
distilling it. It is also simply stronger -- in a seat-balanced 560-game
tournament it took 53.1% against a 25% parity line, where ASU took 34.1% and our
best trained agent 32.2%.

The economics are what make this viable where ASU distillation was not. Kuzey
decides in ~0.07 ms against ASU's 57 ms, so labels cost essentially the price of
simulating the game. That also makes a DAgger round cheap, which matters: plain
behaviour cloning fails on compounding distribution shift, because the student's
own mistakes lead to states the teacher never demonstrated.

Two subcommands:

  collect  play games with Kuzey in one or more seats and record every decision
           it made -- state, the legal set, and its choice.
  train    masked cross-entropy on those decisions, saved as a format-3 PPO
           checkpoint so every existing evaluator loads it unchanged.

Forced states (one legal action) are skipped: they carry no preference
information and would dominate the loss with free accuracy.

The student is saved with ``hybrid=False`` on purpose. Under ``hybrid=True`` the
evaluator intercepts BUY_PROPERTY and ACCEPT_TRADE with fixed rules and strips
them from the allowed set, which would throw away Kuzey's judgment on exactly
those decisions. Kuzey is better at them than the fixed rules are.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.state import STATE_DIM  # noqa: E402

NUM_PLAYERS = 4

# The fields are chosen for state coverage rather than difficulty: the student
# must behave sensibly in positions produced by weak, strong and neural play,
# because on match day the other three seats are unknown agents.
#
# An earlier version of this file excluded ASU entirely, "as an opponent or
# otherwise". That was a mistake and it cost us a whole training round. The rule
# forbids *imitating* ASU -- using it as a training opponent is explicitly
# allowed, and the rest of this project already does. Excluding it meant the
# student never saw a single board state an ASU opponent had produced, and it
# then scored 14-21% in ASU fields against 39-45% everywhere else. That was a
# hole in the data, not a weakness of the method.
#
# Kuzey remains the only labeller. ASU only ever occupies an opponent seat, so
# no ASU decision is ever a training target.
FIELDS = [
    ("kuzey", "kuzey", "kuzey"),
    ("kuzey", "kuzey", "fixed-d"),
    ("kuzey", "fixed-a", "fixed-b"),
    ("kuzey", "fixed-e", "fixed-f"),
    ("fixed-a", "fixed-b", "fixed-c"),
    ("fixed-d", "fixed-e", "fixed-f"),
    ("fixed-b", "fixed-d", "fixed-f"),
    ("fixed-c", "fixed-c", "fixed-e"),
    ("kuzey-plus", "fixed-c", "fixed-d"),
    ("kuzey-plus", "kuzey", "fixed-b"),
]
NEURAL_SLOTS = ["CHAMPION.pt", "LAST_RESORT.pt"]

# Collected separately and budgeted separately: an ASU seat costs ~40 core-s per
# game against ~1 s for everything else, so these are counted in games, not
# mixed in by ratio.
ASU_FIELDS = [
    ("asu-value-v1", "fixed-d", "fixed-b"),
    ("asu-value-v1", "kuzey", "fixed-e"),
    ("asu-value-v1", "fixed-c", "fixed-f"),
    ("asu-value-v1", "kuzey", "kuzey"),
    ("asu-value-v1", "asu-value-v1", "fixed-c"),
]
# One ASU seat per field, with a single two-seat field for coverage. A
# three-ASU field was measured at ~5 min of wall clock for one game, which buys
# the same kind of state at three times the price -- what matters here is that
# ASU-influenced positions appear at all, not that they saturate the table.


def _fields_with_neural() -> list[tuple[str, ...]]:
    """Add fields containing our own networks, if those checkpoints exist.

    Neural opposition is the field we historically score worst in and the one
    match day actually consists of, so it belongs in the coverage mix.
    """
    fields = list(FIELDS)
    for rel in NEURAL_SLOTS:
        path = REPO / "artifacts" / rel
        if path.exists():
            fields.append(("kuzey", f"ppo:{path}", "fixed-d"))
            fields.append((f"ppo:{path}", "fixed-b", "fixed-e"))
    return fields


def _play(args):
    """One game. Returns Kuzey's decisions as (state, legal, action) triples."""
    seed, field = args
    from monopoly_game_engine.train import build_opponents

    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=200)
    env.reset()

    # Kuzey occupies the seats named 'kuzey'; rotating by seed keeps its seat
    # from correlating with the observation, whose deed slots are indexed by
    # physical player id.
    rot = seed % NUM_PLAYERS
    ids = list(field)
    ids.insert(rot, "kuzey")
    agents = build_opponents(ids, list(range(NUM_PLAYERS)))
    # Exactly "kuzey", never "kuzey-plus": ChampionPlus is a different policy,
    # and mixing two teachers gives the student contradictory targets for
    # identical states. kuzey-plus stays, but only as an opponent.
    teacher_seats = {i for i, name in enumerate(ids) if name == "kuzey"}

    states, actions, legals = [], [], []
    teacher_illegal = [0]
    decisions = 0
    limit = env.max_rounds * NUM_PLAYERS * 30
    while not env.done and decisions < limit:
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [0]
        action = agents[pid].choose_action(env)
        # The scripted personalities occasionally name an action that is not
        # legal in the current phase; the repo's drivers absorb this the same
        # way (evaluate.py::_ScriptedAdapter). Teacher seats are recorded only
        # when their own choice was legal, so a fallback never becomes a label.
        if action not in allowed:
            if pid in teacher_seats:
                teacher_illegal[0] += 1
            action = (int(ActionType.END_TURN)
                      if int(ActionType.END_TURN) in allowed else allowed[0])
        elif pid in teacher_seats and len(allowed) > 1:
            # Only record real choices. A forced state teaches nothing and
            # would inflate accuracy with free correct answers.
            states.append(np.asarray(env._get_state(pid), dtype=np.float16))
            actions.append(action)
            legals.append(np.asarray(allowed, dtype=np.int16))
        env.step(action)
        decisions += 1
    return states, actions, legals, env.round, not env.done, teacher_illegal[0]


def _play_dagger(args):
    """One DAgger game: the student drives, Kuzey labels what it would have done.

    Behaviour cloning is trained on the teacher's own state distribution, but at
    play time the student visits states its mistakes created, which the teacher
    never demonstrated -- errors compound. DAgger closes that loop by labelling
    the *student's* states with the teacher's answer.

    Only the student's seat is labelled, and the label is Kuzey's choice in that
    exact position, so the data stays teacher-generated.
    """
    seed, field, student = args
    from monopoly_game_engine.train import KuzeyHeuristicOpponent, build_opponents

    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=200)
    env.reset()

    rot = seed % NUM_PLAYERS
    ids = list(field)
    ids.insert(rot, f"ppo:{student}")
    agents = build_opponents(ids, list(range(NUM_PLAYERS)))
    seat = rot
    oracle = KuzeyHeuristicOpponent(seat, "champion")

    states, actions, legals = [], [], []
    agree = seen = 0
    decisions = 0
    limit = env.max_rounds * NUM_PLAYERS * 30
    while not env.done and decisions < limit:
        pid = env.whose_turn()
        if env.players[pid].bankrupt:
            env._advance_turn()
            continue
        allowed = env.get_allowed_actions(pid)
        if not allowed:
            allowed = [0]
        label = None
        if pid == seat and len(allowed) > 1:
            teach = oracle.choose_action(env)
            if teach in allowed:
                label = teach
                states.append(np.asarray(env._get_state(pid), dtype=np.float16))
                actions.append(teach)
                legals.append(np.asarray(allowed, dtype=np.int16))
        action = agents[pid].choose_action(env)
        if action not in allowed:
            action = (int(ActionType.END_TURN)
                      if int(ActionType.END_TURN) in allowed else allowed[0])
        if label is not None:
            seen += 1
            agree += int(action == label)
        env.step(action)
        decisions += 1
    return states, actions, legals, env.round, not env.done, (agree, seen)


def dagger(student: Path, n_games: int, seed_base: int, workers: int, out: Path,
           asu_games: int = 0):
    fields = _fields_with_neural()
    jobs = [(seed_base + i, fields[i % len(fields)], str(student))
            for i in range(n_games)]
    # The student's compounding errors in ASU games are exactly what needs
    # correcting, so ASU fields belong here too, not just in phase 1.
    jobs = [(seed_base + 500000 + i, ASU_FIELDS[i % len(ASU_FIELDS)], str(student))
            for i in range(asu_games)] + jobs
    n_games += asu_games
    print(f"DAgger: {n_games} games driven by {student.name}, labelled by Kuzey",
          flush=True)
    S, A, L, G = [], [], [], []
    ag = sn = 0
    started = time.perf_counter()
    with mp.Pool(workers) as pool:
        for i, (s, a, l, rnd, tr, (x, y)) in enumerate(
                pool.imap_unordered(_play_dagger, jobs, chunksize=4), 1):
            S.extend(s); A.extend(a); L.extend(l); G.extend([i] * len(a))
            ag += x; sn += y
            if i % 200 == 0 or i == n_games:
                el = time.perf_counter() - started
                print(f"  {i}/{n_games} games | {len(A)} labels | "
                      f"student agrees {100*ag/max(sn,1):.1f}% | "
                      f"{i/el:.1f} games/s", flush=True)
    off = np.zeros(len(L) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(x) for x in L])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        states=np.stack(S), actions=np.asarray(A, dtype=np.int16),
        legal_flat=np.concatenate(L), legal_off=off,
        game_id=np.asarray(G, dtype=np.int32),
    )
    print(f"\n{len(A)} labels; agreement on the student's OWN states: "
          f"{100*ag/max(sn,1):.1f}%")
    print(f"-> {out} ({out.stat().st_size/1e6:.0f} MB)")


def collect(n_games: int, seed_base: int, workers: int, out: Path,
            asu_games: int = 0):
    fields = _fields_with_neural()
    jobs = [(seed_base + i, fields[i % len(fields)]) for i in range(n_games)]
    # ASU jobs go first: they are ~40x slower, so starting them last would leave
    # the pool draining a long tail on idle workers.
    jobs = [(seed_base + 500000 + i, ASU_FIELDS[i % len(ASU_FIELDS)])
            for i in range(asu_games)] + jobs
    print(f"collecting {n_games} fast + {asu_games} ASU-opponent games "
          f"over {len(fields)}+{len(ASU_FIELDS)} fields on {workers} workers",
          flush=True)
    n_games += asu_games

    S, A, L, G = [], [], [], []
    rounds = trunc = till = 0
    parts: list[Path] = []
    started = time.perf_counter()
    last_flush = started

    def flush():
        """Write what we have so far as its own shard, then drop it.

        Writing only at the end means a reclaimed Colab session costs the whole
        run rather than the last chunk -- which has already happened once here.
        It also caps peak memory, since a full run holds millions of decisions
        in Python lists. ``train`` takes any number of shards, so parts need no
        reassembly.
        """
        if not A:
            return
        off = np.zeros(len(L) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(x) for x in L])
        part = out.with_name(f"{out.stem}.part{len(parts):02d}.npz")
        part.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(part, states=np.stack(S),
                            actions=np.asarray(A, dtype=np.int16),
                            legal_flat=np.concatenate(L), legal_off=off,
                            game_id=np.asarray(G, dtype=np.int32))
        parts.append(part)
        print(f"    wrote {part.name} ({len(A)} labels, "
              f"{part.stat().st_size/1e6:.0f} MB)", flush=True)
        S.clear(); A.clear(); L.clear(); G.clear()

    with mp.Pool(workers) as pool:
        for i, (s, a, l, rnd, tr, ti) in enumerate(pool.imap_unordered(_play, jobs, chunksize=4), 1):
            S.extend(s); A.extend(a); L.extend(l); G.extend([i] * len(a))
            rounds += rnd; trunc += int(tr); till += ti
            if i % 200 == 0 or i == n_games:
                el = time.perf_counter() - started
                print(f"  {i}/{n_games} games | {len(A)} labels in buffer | "
                      f"{i/el:.1f} games/s | ~{(n_games-i)/(i/el)/60:.1f} min left",
                      flush=True)
            # Flush on games *or* on elapsed time. ASU games run ~200x slower
            # than the rest, so a game-count trigger alone can leave an hour of
            # the most expensive data sitting in memory.
            if i % 2000 == 0 or time.perf_counter() - last_flush > 600:
                flush()
                last_flush = time.perf_counter()
    flush()
    if parts:
        print(f"\n{len(parts)} shards written: {parts[0].parent}/{out.stem}.part*.npz")
        return

    off = np.zeros(len(L) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(x) for x in L])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        states=np.stack(S) if S else np.zeros((0, STATE_DIM), np.float16),
        actions=np.asarray(A, dtype=np.int16),
        legal_flat=np.concatenate(L) if L else np.zeros(0, np.int16),
        legal_off=off,
        game_id=np.asarray(G, dtype=np.int32),
    )
    el = time.perf_counter() - started
    print(f"\n{len(A)} labels from {n_games} games ({len(A)/n_games:.0f}/game), "
          f"{rounds/n_games:.0f} rounds/game, {100*trunc/n_games:.1f}% truncated")
    print(f"teacher illegal actions (excluded from labels): {till}")
    print(f"{el/60:.1f} min, {n_games/el:.1f} games/s -> {out} "
          f"({out.stat().st_size/1e6:.0f} MB)")


def train(shards: list[Path], out: Path, hidden: int, epochs: int,
          batch: int, lr: float, hybrid: bool):
    import torch
    import torch.nn.functional as F
    from monopoly_game_engine.agent_ppo import PPOAgent

    S, A, LF, LO, G = [], [], [], [], []
    base_off = 0
    game_shift = 0
    for sh in shards:
        d = np.load(sh)
        S.append(d["states"]); A.append(d["actions"])
        LF.append(d["legal_flat"]); LO.append(d["legal_off"][1:] + base_off)
        base_off += len(d["legal_flat"])
        G.append(d["game_id"] + game_shift); game_shift = G[-1].max() + 1
    states = np.concatenate(S); actions = np.concatenate(A)
    legal_flat = np.concatenate(LF)
    legal_off = np.concatenate([[0], np.concatenate(LO)])
    games = np.concatenate(G)
    n = len(actions)
    print(f"{n} labels from {len(shards)} shards, {games.max()} games")

    # Split by game, never by row: consecutive decisions inside one game are
    # highly correlated, so a row split leaks trajectories into validation and
    # reports an accuracy the student does not have.
    rng = np.random.default_rng(0)
    uniq = np.unique(games)
    rng.shuffle(uniq)
    val_games = set(uniq[: max(1, len(uniq) // 10)].tolist())
    is_val = np.fromiter((g in val_games for g in games), bool, n)
    tr_idx = np.flatnonzero(~is_val)
    va_idx = np.flatnonzero(is_val)
    print(f"train {len(tr_idx)} / val {len(va_idx)} labels "
          f"({len(uniq)-len(val_games)}/{len(val_games)} games)")

    agent = PPOAgent(player_id=0, hybrid=hybrid, device="cpu",
                     hidden_dim=hidden, restrict_liquidation=False)
    actor = agent.actor
    opt = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    st = torch.from_numpy(states.astype(np.float32))
    ac = torch.from_numpy(actions.astype(np.int64))

    lengths = np.diff(legal_off)

    def build_mask(idx):
        """Scatter the variable-length legal sets into a dense bool mask.

        Vectorised rather than a per-row Python loop: at millions of labels x
        a dozen epochs, the loop costs more than the forward and backward pass
        combined.
        """
        counts = lengths[idx]
        rows = np.repeat(np.arange(len(idx)), counts)
        cols = np.concatenate([legal_flat[legal_off[i]:legal_off[i + 1]] for i in idx]) \
            if len(idx) else np.zeros(0, np.int16)
        m = torch.zeros(len(idx), ACTION_SPACE_SIZE, dtype=torch.bool)
        m[torch.from_numpy(rows), torch.from_numpy(cols.astype(np.int64))] = True
        return m

    best = -1.0
    for ep in range(epochs):
        actor.train()
        perm = np.random.permutation(tr_idx)
        tot = cor = 0
        loss_sum = 0.0
        for b in range(0, len(perm), batch):
            idx = perm[b:b + batch]
            mask = build_mask(idx)
            logits = actor(st[idx], mask)
            logits = logits.masked_fill(~mask, -1e9)
            loss = F.cross_entropy(logits, ac[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss) * len(idx)
            cor += int((logits.argmax(1) == ac[idx]).sum()); tot += len(idx)
        sched.step()

        actor.eval()
        vc = vt = 0
        with torch.inference_mode():
            for b in range(0, len(va_idx), 4096):
                idx = va_idx[b:b + 4096]
                mask = build_mask(idx)
                lg = actor(st[idx], mask).masked_fill(~mask, -1e9)
                vc += int((lg.argmax(1) == ac[idx]).sum()); vt += len(idx)
        va = vc / max(vt, 1)
        print(f"  epoch {ep+1}/{epochs} loss {loss_sum/tot:.4f} "
              f"train {100*cor/tot:.1f}% val {100*va:.1f}%", flush=True)
        if va > best:
            best = va
            agent.save(str(out))
            print(f"    saved (best val {100*best:.1f}%)", flush=True)

    print(f"\nbest held-out top-1 agreement with Kuzey: {100*best:.1f}%")
    print(f"checkpoint: {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--games", type=int, default=2000)
    c.add_argument("--seed-base", type=int, default=100000)
    c.add_argument("--workers", type=int, default=10)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--asu-games", type=int, default=0,
                   help="games with ASU in opponent seats (Kuzey still labels)")
    g = sub.add_parser("dagger")
    g.add_argument("--student", type=Path, required=True)
    g.add_argument("--games", type=int, default=3000)
    g.add_argument("--seed-base", type=int, default=700000)
    g.add_argument("--workers", type=int, default=10)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--asu-games", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--shards", type=Path, nargs="+", required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--hybrid", action="store_true")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.games, a.seed_base, a.workers, a.out, a.asu_games)
    elif a.cmd == "dagger":
        dagger(a.student, a.games, a.seed_base, a.workers, a.out, a.asu_games)
    else:
        train(a.shards, a.out, a.hidden, a.epochs, a.batch, a.lr, a.hybrid)


if __name__ == "__main__":
    main()
