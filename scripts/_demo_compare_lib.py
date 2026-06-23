"""Shared loader + behavioural-signature helpers for comparing sim-generated demos
against recorded (real-hardware) demos. Used by the orchestrated demo-validation workflow
so each analysis agent does NOT re-solve the LeRobotDataset parquet layout.

Handles both v3.0 layouts: one-file-per-episode (file-NNN.parquet) and all-episodes-in-one-file
(grouped by episode_index). Returns per-episode numpy arrays.

NOTE on units (do NOT compare raw values across datasets):
  recorded so101-pickplace-new : action/state in SERVO DEGREES; state[12]=joint_pos[0:6]+joint_vel[6:12];
                                 action[6]=joint_0..5 (action[5]=GRIPPER).
  sim    so101-sim-pickplace-* : action normalized deltas (gripper +1 open / -1 close);
                                 state[13]=joint_pos_rel[0:6]+object_pose[6:13] (state[5]=gripper joint).
Compare PATTERNS (per-episode-normalized), ranges, smoothness — never raw magnitudes.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq


def load_episodes(dataset_dir: str):
    """Return dict {episode_index: {"state": (T,D) float32, "action": (T,A) float32}}.
    Episodes are sorted by frame_index within each episode."""
    files = sorted(glob.glob(os.path.join(dataset_dir, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset_dir}/data")
    eps: dict[int, dict[str, list]] = {}
    for f in files:
        df = pq.read_table(f).to_pandas()
        if "episode_index" not in df.columns:
            continue
        for ep, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index") if "frame_index" in g.columns else g
            st = np.stack([np.asarray(x, dtype=np.float32) for x in g["observation.state"].to_numpy()])
            ac = np.stack([np.asarray(x, dtype=np.float32) for x in g["action"].to_numpy()])
            eps.setdefault(int(ep), {"state": [], "action": []})
            eps[int(ep)]["state"].append(st)
            eps[int(ep)]["action"].append(ac)
    out = {}
    for ep, d in eps.items():
        out[ep] = {
            "state": np.concatenate(d["state"], axis=0),
            "action": np.concatenate(d["action"], axis=0),
        }
    return out


def gripper_series(ep_arr: np.ndarray, col: int = 5) -> np.ndarray:
    """Gripper channel (action[:,5] by default), min-max normalized to [0,1] per episode.
    Returns NaN-free 1-D array; flat episodes -> all zeros."""
    g = ep_arr[:, col].astype(np.float64)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-9:
        return np.zeros_like(g)
    return (g - lo) / (hi - lo)


def grasp_pattern(gnorm: np.ndarray, close_is_low: bool):
    """Characterize a normalized gripper series as a single grasp:
    close once mid-episode, hold, open near the end. Returns dict with the close/open
    fractional timestamps and a boolean `clean_single_grasp`.
    `close_is_low`: True if a CLOSED gripper is the LOW end of the normalized range."""
    T = len(gnorm)
    if T < 5:
        return {"clean_single_grasp": False, "n_transitions": 0, "close_t": None, "open_t": None}
    closed = (gnorm < 0.5) if close_is_low else (gnorm > 0.5)  # bool per step (is the gripper closed)
    # count transitions open<->closed
    trans = int(np.sum(closed[1:] != closed[:-1]))
    close_idx = np.argmax(closed) if closed.any() else None  # first closed step
    # last closed step
    open_after = None
    if closed.any():
        last_closed = len(closed) - 1 - int(np.argmax(closed[::-1]))
        open_after = last_closed
    return {
        "clean_single_grasp": bool(closed.any() and trans <= 3),  # close + (hold) + open ~= 1-2 transitions
        "n_transitions": trans,
        "closed_fraction": float(closed.mean()),
        "close_t": (float(close_idx) / T) if close_idx is not None else None,
        "open_t": (float(open_after) / T) if open_after is not None else None,
    }


def spectral_arc_length(traj: np.ndarray, fs: float = 30.0) -> float:
    """SAL smoothness of a 1-D trajectory (more negative = smoother). Standard SPARC-lite:
    arc length of the normalized magnitude spectrum of the speed profile."""
    v = np.diff(traj.astype(np.float64))
    if len(v) < 4 or np.allclose(v, 0):
        return 0.0
    V = np.abs(np.fft.rfft(v))
    if V.max() < 1e-12:
        return 0.0
    Vn = V / V.max()
    f = np.linspace(0, 0.5 * fs, len(Vn))
    # arc length of (f, Vn) normalized by frequency span
    df = np.diff(f) / (f[-1] - f[0] + 1e-12)
    dV = np.diff(Vn)
    return -float(np.sum(np.sqrt(df**2 + dV**2)))


if __name__ == "__main__":  # quick self-test on whatever exists
    import sys

    for d in sys.argv[1:]:
        eps = load_episodes(d)
        ks = sorted(eps)
        T = [eps[k]["state"].shape[0] for k in ks]
        sdim = eps[ks[0]]["state"].shape[1]
        adim = eps[ks[0]]["action"].shape[1]
        print(f"{d}: {len(ks)} eps, state_dim={sdim} action_dim={adim} "
              f"len[min={min(T)} max={max(T)} mean={np.mean(T):.0f}]")
        # gripper sanity on ep 0
        a0 = eps[ks[0]]["action"]
        print(f"   ep{ks[0]} action[:,5] range=[{a0[:,5].min():.3f},{a0[:,5].max():.3f}]")
