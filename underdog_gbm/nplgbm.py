"""Evaluate a LightGBM text model with numpy alone.

Why this exists
---------------
LightGBM ships a compiled extension, ``lib_lightgbm.so``, which links
``libgomp.so.1`` dynamically. The match image is ``python:3.12-slim-bookworm``
and does not carry the OpenMP runtime, the image is built with
``--network=none`` so nothing can be installed at build time, and the manylinux
wheel does not vendor libgomp despite what our requirements.txt used to claim.
The result was ``OSError: libgomp.so.1: cannot open shared object file`` at
``import lightgbm``, so the submitted policy fell back to the rule agent in
every game while looking healthy from the outside.

The boosters are plain text and the trees are plain numeric splits, so the
dependency is avoidable rather than negotiable. This parses the two model files
and walks them in numpy, which the agent already depends on.

Scope, checked against the shipped models
-----------------------------------------
* ``decision_type`` is 2 for every split in both models: no categorical splits,
  no linear trees, missing-type None. The rule is therefore exactly
  ``value <= threshold -> left``, with no missing-value branch to reproduce.
* ``num_class=1`` and no ``average_output``, so the raw score is the sum of one
  leaf value per tree. Shrinkage is already folded into the leaf values.
* Only the ranking *order* is used by the policy, which takes an argmax, but
  the scores match LightGBM's own to within floating-point noise anyway.

If a future model uses categorical splits or missing types, ``_parse_tree``
raises rather than silently scoring it wrong.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["NumpyBooster"]

_MAX_DEPTH = 256          # a walk this deep means a malformed tree, not a deep one


def _floats(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _ints(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.float64).astype(np.int32)


class _Tree:
    """One tree, flattened into arrays indexed by internal-node id.

    A child id is a node id when non-negative and encodes a leaf as
    ``-(leaf_index + 1)`` when negative, which is LightGBM's own convention.
    """

    __slots__ = ("split_feature", "threshold", "left", "right", "leaf_value",
                 "constant")

    def __init__(self, block: dict[str, str]) -> None:
        leaves = int(block["num_leaves"])
        self.leaf_value = _floats(block["leaf_value"])
        if leaves <= 1:
            # A stump: no splits, one value for every row.
            self.constant = float(self.leaf_value[0]) if self.leaf_value.size else 0.0
            self.split_feature = self.threshold = None
            self.left = self.right = None
            return
        self.constant = None

        decision = _ints(block["decision_type"])
        if decision.size and int(np.max(decision)) > 2:
            raise ValueError(
                "model uses categorical splits or a missing-value type this "
                f"evaluator does not implement (decision_type={sorted(set(decision.tolist()))})")

        self.split_feature = _ints(block["split_feature"])
        self.threshold = _floats(block["threshold"])
        self.left = _ints(block["left_child"])
        self.right = _ints(block["right_child"])

    def predict(self, X: np.ndarray, out: np.ndarray) -> None:
        """Add this tree's contribution for every row of ``X`` into ``out``."""
        if self.constant is not None:
            out += self.constant
            return
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int32)
        rows = np.arange(n)
        for _ in range(_MAX_DEPTH):
            live = node >= 0
            if not live.any():
                break
            idx = node[live]
            values = X[rows[live], self.split_feature[idx]]
            go_left = values <= self.threshold[idx]
            node[live] = np.where(go_left, self.left[idx], self.right[idx])
        else:  # pragma: no cover - only a malformed model gets here
            raise ValueError("tree walk did not terminate")
        out += self.leaf_value[-node - 1]


class NumpyBooster:
    """Drop-in replacement for the two calls this agent makes on a Booster."""

    def __init__(self, model_file: str | Path) -> None:
        self.path = str(model_file)
        self.trees: list[_Tree] = []
        self.num_feature = 0
        self._load()

    def _load(self) -> None:
        text = Path(self.path).read_text()
        head, _, rest = text.partition("Tree=")
        for line in head.splitlines():
            if line.startswith("max_feature_idx="):
                self.num_feature = int(line.split("=", 1)[1]) + 1
        if not rest:
            raise ValueError(f"no trees in {self.path}")

        body = ("Tree=" + rest).split("end of trees")[0]
        for chunk in body.split("Tree=")[1:]:
            block: dict[str, str] = {}
            for line in chunk.splitlines():
                key, sep, value = line.partition("=")
                if sep:
                    block[key.strip()] = value.strip()
            if "num_leaves" in block and "leaf_value" in block:
                self.trees.append(_Tree(block))
        if not self.trees:
            raise ValueError(f"parsed no usable trees from {self.path}")

    def predict(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees:
            tree.predict(X, out)
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"NumpyBooster({Path(self.path).name}, trees={len(self.trees)})"
