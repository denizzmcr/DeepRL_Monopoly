# Running champion v2 training on a Colab TPU VM

The TPU itself is not used. The workload is Python game simulation plus a 2.5M
parameter MLP, and neither wants a TPU: `torch_xla` compiles static graphs, our
action masks change shape every step, and the resulting recompilation is slower
than CPU. What makes the TPU runtime worth having is its **host**: 44 vCPUs and
172 GB against this laptop's 12 cores.

That matters most for the ASU league. ASU costs ~55 s/game against ~1 s for
every other opponent, and collection happens in synchronous rounds, so a round
waits on its slowest game. On 4 workers here that run manages 0.18 games/s. With
40 workers many more ASU games run concurrently and the barrier hurts far less.

Paste each cell into a Colab notebook on the TPU runtime, in order.

---

## 1. Check the host

```python
import os
print("vCPUs:", os.cpu_count())
!free -g | head -2
```

Expect 44 vCPUs. If it is 8, stop — this laptop is faster.

## 2. Clone and install

```python
!git clone -b feat/asu-shards https://github.com/denizzmcr/DeepRL_Monopoly.git
%cd DeepRL_Monopoly
!pip -q install torch numpy
```

The repo is public, so no token is needed. `feat/asu-shards` carries the league,
the parallel trainer, `artifacts/CHAMPION.pt`, and the two checkpoints the league
uses as opponents.

## 3. Persist checkpoints to Drive

Colab disconnects after ~12 hours and on idle. Training snapshots every 500
games, so a disconnect costs at most 500 games — but only if the snapshots are
somewhere that survives the session.

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/monopoly_champion_v2
!ln -sfn /content/drive/MyDrive/monopoly_champion_v2 artifacts/diag/champion_v2
```

## 4. Train

The ASU league, which is the run that benefits. 40 workers leaves 4 cores for
the learner and the OS. A large round keeps workers busy while slow ASU games
finish.

```python
!WORKERS=40 ROUND=120 python tools/train_champion_v2.py 512 asu
```

For a second candidate at higher capacity, in a second session:

```python
!WORKERS=40 ROUND=120 python tools/train_champion_v2.py 1024 asu
```

## 5. Watch it

The first line reports the opponent plan; confirm `asu-value-v1` appears in it.
Then each 500-game window prints:

```
Game  1500 | Win%: 30.6% | 1.80 games/s | collect 12.1s update 3.2s
```

`collect` against `update` says where the time goes. On this laptop the ASU run
sits at `collect 40.5s update 2.5s` — collection dominates by 16x, which is
exactly what more cores fixes.

## What to bring back

Checkpoints land in Drive as `champion_v2_h512_asu_g000500.pt` and so on. Copy
them into `artifacts/diag/champion_v2/` on the main machine, then screen them
against the other candidates on fields they never trained against, and select on
worst case rather than on the training win rate — training curves have misled
this project twice.

## What not to bother with

- **The TPU.** Nothing here uses it.
- **A GPU runtime.** The update is ~12 s against ~2 s of collection in the fast
  league and would benefit, but the ASU league is collection-bound by 16x, so
  cores are worth more than a GPU for the run we are moving.
- **`torch_xla`.** Dynamic action-mask shapes would recompile constantly.
