"""Supervise the Colab training fleet unattended: sync, watch, and repair.

This replaces colab_sync.py. The sync half is unchanged -- checkpoints live on
the VM, and a session that dies takes with it everything since its last 250-game
save, so they get pulled to the laptop every cycle.

What is new is repair. Nobody is watching these hosts, so the two failure modes
that actually end a run have to be handled here:

* **An agent process dies** (OOM, a raised exception in the learner). The host
  stays up and the other agents keep going, so nothing looks wrong from outside
  -- the log simply stops. Detected by absence from the process list, and fixed
  by relaunching from the agent's own latest checkpoint.
* **An agent wedges** without dying. Detected by its log not growing across
  STALL_CYCLES cycles. The threshold is deliberately generous: a single ASU game
  can run 10 minutes and a round waits on its slowest game, so a live agent can
  legitimately be quiet for a long time. Restarting a healthy-but-slow agent
  throws away real work, so the bias here is toward waiting.

A session that cannot be reached at all is reported, not repaired -- recreating
it needs an accelerator assignment that may not be available, and that is a
decision for a human.
"""
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path("/Users/denizmacbook/DeepRL_Monopoly")
LOCAL = REPO / "artifacts" / "diag" / "champion_v2"
REMOTE = "/content/DeepRL_Monopoly/artifacts/diag/champion_v2"
SCRATCH = Path("/private/tmp/claude-501/-Users-denizmacbook-DeepRL-Monopoly/"
               "8c901f93-e8ed-4515-be6a-20f286e5eed8/scratchpad")
REFRESH = SCRATCH / "colab_refresh.py"
INTERVAL = 900            # 15 minutes
STALL_CYCLES = 6          # 90 minutes of a log not growing before we act

# The fleet, as launched. Every field here is needed to rebuild an agent
# identically after a crash: width and seed define the run, workers and
# games_per_round define its throughput.
FLEET = {
    "c1": dict(workers=14, gpr=42, agents=[
        ("h512_a1", 512, 508), ("h512_a2", 512, 509), ("h512_a3", 512, 510)]),
    "c2": dict(workers=14, gpr=42, agents=[
        ("h1024_a1", 1024, 1008), ("h1024_a2", 1024, 1009), ("h1024_a3", 1024, 1010)]),
    "c3": dict(workers=11, gpr=33, agents=[
        ("h512_b1_c3", 512, 7001), ("h512_b2_c3", 512, 7002)]),
    "c4": dict(workers=11, gpr=33, agents=[
        ("h1024_b1_c4", 1024, 8001), ("h1024_b2_c4", 1024, 8002)]),
}

LOCAL.mkdir(parents=True, exist_ok=True)
state = {}        # tag -> {"size": int, "quiet": int}
unreachable = {}  # session -> consecutive failures


def sh(cmd, timeout=600):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "", "timeout")


def say(msg):
    print(f"[{time.strftime('%m-%d %H:%M')}] {msg}", flush=True)


PROBE = r'''
import glob, os, subprocess
d = "@@REMOTE@@"
print("FILES " + ",".join(sorted(os.path.basename(p) for p in glob.glob(d + "/*.pt"))))
for log in sorted(glob.glob("/content/log_*.txt")):
    tag = os.path.basename(log)[4:-4]
    try:
        lines = [l for l in open(log) if "Game " in l]
        last = lines[-1].strip() if lines else "(no window yet)"
        print("PROG %s|%d|%s" % (tag, os.path.getsize(log), last))
    except Exception:
        print("PROG %s|0|unreadable" % tag)
# A parent whose workers have all died still shows up in the process list, so
# report both: the tags that are alive and the total worker count.
out = subprocess.run("ps -eo args | grep -o 'run_[A-Za-z0-9_]*\\.py' | sort -u",
                     shell=True, capture_output=True, text=True).stdout
print("ALIVE " + ",".join(sorted(x[4:-3] for x in out.split())))
print("WORKERS " + (subprocess.run("pgrep -fc spawn_main", shell=True,
      capture_output=True, text=True).stdout.strip() or "0"))
'''.replace("@@REMOTE@@", REMOTE)

# Rebuilds one agent exactly as launched, resuming from its own checkpoint.
# The __main__ guard matters: spawned workers re-import this file, and without
# it each one re-enters training and exits with "0 games collected".
LAUNCH = r'''
import os, subprocess, textwrap
runner = "/content/run_@@TAG@@.py"
open(runner, "w").write(textwrap.dedent("""
    import sys
    from pathlib import Path
    REPO = Path("/content/DeepRL_Monopoly")
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
    from monopoly_game_engine import PPOAgent
    from monopoly_game_engine.parallel_ppo import train_parallel
    import train_champion_v2 as t

    def main():
        ckpt = REPO / "artifacts/diag/champion_v2/v2_@@TAG@@.pt"
        agent = PPOAgent(player_id=0, hybrid=True, device="cpu",
                         restrict_liquidation=True, entropy_coef=0.005,
                         hidden_dim=@@HID@@)
        if ckpt.exists():
            agent.load(str(ckpt))
            print("resumed from", ckpt.name, flush=True)
        else:
            print("NO CHECKPOINT -- starting fresh", flush=True)
        print("agent @@TAG@@: hidden=@@HID@@ seed=@@SEED@@ gpr=@@GPR@@", flush=True)
        train_parallel(agent, n_games=200000, workers=@@WORKERS@@,
                       opponents=t.league("asu"), games_per_round=@@GPR@@,
                       seed=@@SEED@@, rotate_seats=True, log_every=100,
                       checkpoint_path=str(ckpt), checkpoint_every=250,
                       keep_snapshots=True)

    if __name__ == "__main__":
        main()
"""))
p = subprocess.Popen(["python", runner],
                     stdout=open("/content/log_@@TAG@@.txt", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True,
                     env=dict(os.environ, PYTHONUNBUFFERED="1"))
print("RELAUNCHED @@TAG@@ pid", p.pid)
'''

probe_file = Path("/tmp/sup_probe.py")
probe_file.write_text(PROBE)


def relaunch(session, tag, hid, seed, workers, gpr):
    """Restart one dead agent from its latest checkpoint."""
    body = (LAUNCH.replace("@@TAG@@", tag).replace("@@HID@@", str(hid))
                  .replace("@@SEED@@", str(seed)).replace("@@WORKERS@@", str(workers))
                  .replace("@@GPR@@", str(gpr)))
    f = Path(f"/tmp/sup_launch_{tag}.py")
    f.write_text(body)
    r = sh(f"colab exec -s {session} -f {f}", timeout=900)
    ok = "RELAUNCHED" in (r.stdout or "")
    say(f"  !! [{session}] {tag} {'relaunched' if ok else 'RELAUNCH FAILED'}")
    return ok


def keep_awake():
    """A sleeping laptop stops the sync and lets idle Colab sessions die."""
    if not sh("pgrep -x caffeinate", timeout=30).stdout.strip():
        subprocess.Popen(["caffeinate", "-dimsu"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        say("  !! caffeinate was gone; restarted it")


say(f"supervisor up: {sum(len(v['agents']) for v in FLEET.values())} agents "
    f"across {len(FLEET)} hosts, {INTERVAL // 60} min cycle")

while True:
    try:
        keep_awake()
        sh(f"{REPO}/venv/bin/python {REFRESH}", timeout=180)

        for session, cfg in FLEET.items():
            r = sh(f"colab exec -s {session} -f {probe_file}", timeout=600)
            out = r.stdout or ""
            if "ALIVE" not in out:
                unreachable[session] = unreachable.get(session, 0) + 1
                say(f"  ** [{session}] UNREACHABLE "
                    f"({unreachable[session]} cycles) -- cannot repair from here")
                continue
            unreachable[session] = 0

            remote, alive, workers, progress = [], set(), "?", {}
            for line in out.splitlines():
                if line.startswith("FILES "):
                    remote = [x for x in line[6:].split(",") if x]
                elif line.startswith("ALIVE "):
                    alive = {x for x in line[6:].split(",") if x}
                elif line.startswith("WORKERS "):
                    workers = line[8:].strip()
                elif line.startswith("PROG "):
                    tag, size, last = line[5:].split("|", 2)
                    progress[tag] = (int(size), last)

            say(f"  [{session}] {len(alive)}/{len(cfg['agents'])} agents, "
                f"{workers} workers, {len(remote)} checkpoints")

            for tag, hid, seed in cfg["agents"]:
                size, last = progress.get(tag, (0, "(no log)"))
                st = state.setdefault(tag, {"size": -1, "quiet": 0})

                if tag not in alive:
                    say(f"    [{session}] {tag} DEAD -- {last}")
                    if relaunch(session, tag, hid, seed, cfg["workers"], cfg["gpr"]):
                        st.update(size=-1, quiet=0)
                    continue

                # A log that is not growing is the only stall signal available;
                # the game counter alone is too coarse at 100 games per window.
                st["quiet"] = st["quiet"] + 1 if size == st["size"] else 0
                st["size"] = size
                flag = f" quiet {st['quiet']}c" if st["quiet"] else ""
                say(f"    [{session}] {tag}{flag} | {last}")

                if st["quiet"] >= STALL_CYCLES:
                    say(f"    [{session}] {tag} STALLED "
                        f"{st['quiet'] * INTERVAL // 60} min -- restarting")
                    killer = Path(f"/tmp/sup_kill_{tag}.py")
                    killer.write_text(
                        "import subprocess, time\n"
                        f"subprocess.run('pkill -f run_{tag}.py', shell=True)\n"
                        "time.sleep(5)\nprint('killed')\n")
                    sh(f"colab exec -s {session} -f {killer}", timeout=300)
                    if relaunch(session, tag, hid, seed, cfg["workers"], cfg["gpr"]):
                        st.update(size=-1, quiet=0)

            new = [f for f in remote if not (LOCAL / f).exists()]
            for f in new:
                sh(f"colab download -s {session} {REMOTE}/{f} {LOCAL}/{f}", timeout=900)
                say(f"    {'pulled' if (LOCAL / f).exists() else 'FAILED'} {f}")

    except Exception as exc:
        say(f"cycle error ({type(exc).__name__}: {exc}); continuing")
    time.sleep(INTERVAL)
