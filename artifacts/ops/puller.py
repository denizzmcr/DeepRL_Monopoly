"""Pull Colab checkpoints to the laptop on a short timer. Nothing else.

Colab reclaimed four sessions during this project. Twice that cost real work --
two hours of collection, then a 4.7 hour training run -- and both times the
reason was the same: results existed only on the host. A host is not storage.

This is deliberately smaller than supervisor.py. It does not relaunch anything
or judge whether a run is healthy; it copies files down, and its only job is to
make a lost session cost minutes instead of hours. Run it beside any Colab work
that produces checkpoints.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("/Users/denizmacbook/DeepRL_Monopoly")
LOCAL = REPO / "artifacts" / "diag" / "champion_v2"
REMOTE = "/content/DeepRL_Monopoly/artifacts/diag/champion_v2"
SESSIONS = ["d1", "d2", "d3"]
REFRESH = REPO / "artifacts" / "ops" / "colab_refresh.py"
INTERVAL = 480          # 8 minutes -- the bound on what a lost session costs

LOCAL.mkdir(parents=True, exist_ok=True)

PROBE = f"""
import glob, os
print("FILES " + ",".join(sorted(
    os.path.basename(p) for p in glob.glob("{REMOTE}/*.pt"))))
for f in sorted(glob.glob("/content/*.log")):
    L = [l.rstrip() for l in open(f) if "Game " in l]
    if L:
        print("PROG " + os.path.basename(f) + " " + L[-1].strip())
"""
probe = Path("/tmp/puller_probe.py")
probe.write_text(PROBE)


def sh(cmd, timeout=900):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "", "timeout")


def say(msg):
    print(f"[{time.strftime('%m-%d %H:%M')}] {msg}", flush=True)


say(f"puller up: {SESSIONS}, every {INTERVAL // 60} min -> {LOCAL}")
while True:
    try:
        sh(f"{REPO}/venv/bin/python {REFRESH}", timeout=180)
        # Sequential on purpose: concurrent `colab exec` calls race on
        # sessions.json and have silently renamed hosts mid-run before.
        for s in SESSIONS:
            r = sh(f"colab exec -s {s} -f {probe}", timeout=600)
            out = r.stdout or ""
            if "FILES" not in out:
                say(f"  [{s}] unreachable this cycle")
                continue
            remote = []
            for line in out.splitlines():
                if line.startswith("FILES "):
                    remote = [x for x in line[6:].split(",") if x]
                elif line.startswith("PROG "):
                    say(f"  [{s}] {line[5:]}")
            new = [f for f in remote if not (LOCAL / f).exists()]
            if new:
                say(f"  [{s}] {len(new)} new of {len(remote)}")
            for f in new:
                sh(f"colab download -s {s} {REMOTE}/{f} {LOCAL}/{f}", timeout=900)
                say(f"    {'pulled' if (LOCAL / f).exists() else 'FAILED'} {f}")
    except Exception as exc:
        say(f"cycle error ({type(exc).__name__}: {exc}); continuing")
    time.sleep(INTERVAL)
