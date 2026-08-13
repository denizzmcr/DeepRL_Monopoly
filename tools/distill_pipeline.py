"""Run the whole the heuristic distillation on one host, unattended, start to finish.

Every phase used to be launched by hand, which meant the host sat idle between
them waiting for a human. This runs collect -> train -> DAgger -> retrain as one
process, so a host started at midnight is still working at 05:00.

Failure handling is deliberate rather than optimistic:

* every phase is retried once, because the failures seen in this project were
  transient (a worker dying, a shard write racing a read) rather than logical;
* a phase that fails twice is skipped, and later phases run on whatever data
  already exists -- a failed DAgger round should still leave the phase-2
  student, not lose the night;
* shards are written incrementally by ``collect``/``dagger`` themselves, so a
  reclaimed session costs the last few minutes, not the run. Colab reclaimed
  three sessions during this project and one of them cost two hours.

Phase order matters: the ASU-opponent games come first inside each collection
because they are ~200x slower than the rest, and finishing them last would leave
a long tail draining on otherwise idle workers.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/DeepRL_Monopoly")
TOOL = REPO / "tools" / "distill_heuristic.py"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(args: list[str], phase: str, attempts: int = 2) -> bool:
    for attempt in range(1, attempts + 1):
        say(f"=== {phase} (attempt {attempt}/{attempts}) ===")
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, str(TOOL)] + args,
                              cwd=str(REPO), text=True)
        mins = (time.perf_counter() - started) / 60
        if proc.returncode == 0:
            say(f"=== {phase} OK in {mins:.0f} min ===")
            return True
        say(f"!!! {phase} failed rc={proc.returncode} after {mins:.0f} min")
    say(f"!!! {phase} giving up; continuing with what exists")
    return False


def shards(pattern: str) -> list[str]:
    return sorted(str(p) for p in Path("/content").glob(pattern))


def main() -> None:
    host = sys.argv[1]
    hidden = sys.argv[2]
    seed = int(sys.argv[3])
    workers = sys.argv[4] if len(sys.argv) > 4 else "22"

    say(f"pipeline start: host={host} hidden={hidden} workers={workers}")

    # Phase 1 -- teacher demonstrations, 5% of labels from ASU-opponent games.
    run(["collect", "--games", "3500", "--asu-games", "310",
         "--workers", workers, "--seed-base", str(seed),
         "--out", f"/content/v2shard_{host}.npz"], "phase 1 collect")

    base = shards(f"v2shard_{host}*.npz")
    if not base:
        say("no shards from phase 1; nothing further is possible")
        return
    say(f"phase 1 produced {len(base)} shards")

    student = f"/content/stu_{host}_h{hidden}.pt"
    run(["train", "--shards", *base, "--out", student,
         "--hidden", hidden, "--epochs", "12", "--batch", "512",
         "--lr", "3e-4"], "phase 2 train")

    if not Path(student).exists():
        say("no student checkpoint; stopping")
        return

    # Phase 3 -- DAgger on the student's OWN states, ASU fields included. The
    # student's compounding errors in ASU games are the thing being corrected,
    # so excluding ASU here would repeat the mistake phase 1 already fixed.
    run(["dagger", "--student", student, "--games", "3000",
         "--asu-games", "250", "--workers", workers,
         "--seed-base", str(seed + 700000),
         "--out", f"/content/v2dag_{host}.npz"], "phase 3 dagger")

    every = base + shards(f"v2dag_{host}*.npz")
    say(f"phase 4 training on {len(every)} shards")
    run(["train", "--shards", *every, "--out", f"/content/stu2_{host}_h{hidden}.pt",
         "--hidden", hidden, "--epochs", "8", "--batch", "512",
         "--lr", "3e-4"], "phase 4 retrain")

    say("pipeline complete")
    for p in sorted(Path("/content").glob(f"stu*_{host}_*.pt")):
        say(f"  {p.name}  {p.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
