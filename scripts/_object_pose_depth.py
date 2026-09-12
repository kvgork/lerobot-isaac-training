#!/usr/bin/env python3
"""Depth-based object pose for the SO-101 workspace — the RLVR pose source.

WHY THIS EXISTS
---------------
``binary_success_verifier`` and ``lerobot_isaac_env.outcome_verifier`` have had sound
success predicates for months (``object_in_box``, ``object_lifted``, ...), all of which
consume a plain state dict. Nothing on this rig ever produced that dict, so every
predicate was unreachable on hardware and every episode had to be labelled by an
operator keypress. That single gap blocks automated eval scoring AND the self-improving
collection loop, where an unlabelled round trains on its own failures.

This produces the dict, from depth alone — no fiducials, no colour assumptions, no
learned model to train or drift.

METHOD
------
1. Keep only depth inside a workspace band (the table sits ~0.61 m from this camera;
   the raw frame reaches 31 m and includes the whole room, which is what made a naive
   plane fit lock onto the wrong surface).
2. RANSAC a plane to the table.
3. Points 10-200 mm above that plane are objects.
4. Connected components -> blobs with an image centroid, a 3-D centroid and a height.
5. Classify by size and height priors into cup / die.

MEASURED LIMITS — read before trusting a verdict
------------------------------------------------
The ARM is also above the table and is far larger than either object: observed blobs of
19 000 and 11 000 px against a 16 mm die. Classification therefore only works when the
arm is out of the workspace, which in practice means **observe at episode start (arm at
home) and at episode end (arm retracted)**, not mid-episode.

Plane-fit quality on this rig was 37% inliers of the workspace band. That is adequate for
"is something 5 cm above the table" and NOT adequate for millimetre pose. Treat the
output as a coarse verdict, not a measurement.

Nothing here is validated against ground truth yet. ``--calibrate`` exists so an operator
confirms which blob is which before any of it is trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORKSPACE_NEAR_M = 0.35
WORKSPACE_FAR_M = 0.85
PLANE_TOL_M = 0.005
OBJECT_MIN_H_M = 0.010
OBJECT_MAX_H_M = 0.200
MIN_BLOB_PX = 150


@dataclass
class Blob:
    area_px: int
    centroid_px: tuple[float, float]
    centroid_xyz: tuple[float, float, float]
    height_mm: float
    label: str = "unknown"


@dataclass
class Scene:
    blobs: list[Blob] = field(default_factory=list)
    plane_inlier_frac: float = 0.0

    def by_label(self, label: str) -> Blob | None:
        for b in self.blobs:
            if b.label == label:
                return b
        return None


def capture_depth(warmup: int = 45):
    """Grab one depth frame plus its intrinsics and scale."""
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    try:
        for _ in range(max(1, warmup)):
            frames = pipe.wait_for_frames()
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        intr = (
            prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        )
    finally:
        pipe.stop()
    return depth * scale, intr


def fit_table_plane(pts: np.ndarray, iters: int = 300, seed: int = 0):
    """RANSAC a plane. Returns (normal, d, inlier_fraction) with the normal pointing up."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        p = pts[idx]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ p[0]
        inliers = int((np.abs(pts @ n + d) < PLANE_TOL_M).sum())
        if best is None or inliers > best[0]:
            best = (inliers, n, d)
    inliers, n, d = best
    heights = -(pts @ n + d)
    if np.median(heights) < 0:  # make "up" positive
        n, d = -n, -d
    return n, d, inliers / len(pts)


def detect(depth_m: np.ndarray, intr, min_blob_px: int = MIN_BLOB_PX) -> Scene:
    """Find objects standing above the table."""
    import cv2

    band = (depth_m > WORKSPACE_NEAR_M) & (depth_m < WORKSPACE_FAR_M)
    if band.sum() < 5000:
        return Scene()
    ys, xs = np.nonzero(band)
    Z = depth_m[band]
    X = (xs - intr.ppx) / intr.fx * Z
    Y = (ys - intr.ppy) / intr.fy * Z
    pts = np.stack([X, Y, Z], axis=1)

    n, d, frac = fit_table_plane(pts)
    h = -(pts @ n + d)
    above = (h > OBJECT_MIN_H_M) & (h < OBJECT_MAX_H_M)
    scene = Scene(plane_inlier_frac=frac)
    if above.sum() < 100:
        return scene

    mask = np.zeros(depth_m.shape, np.uint8)
    mask[ys[above], xs[above]] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_lab, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    lab_above = lab[ys[above], xs[above]]
    for i in range(1, n_lab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_blob_px:
            continue
        sel = lab_above == i
        if sel.sum() == 0:
            continue
        scene.blobs.append(
            Blob(
                area_px=area,
                centroid_px=(float(cents[i][0]), float(cents[i][1])),
                centroid_xyz=tuple(float(v) for v in pts[above][sel].mean(axis=0)),
                height_mm=float(np.median(h[above][sel]) * 1000.0),
            )
        )
    scene.blobs.sort(key=lambda b: b.area_px, reverse=True)
    return scene


def classify(scene: Scene, priors: dict) -> Scene:
    """Label blobs cup / die from size+height priors; everything else stays 'unknown'.

    Deliberately conservative: a blob matching neither prior is left unknown rather than
    forced into the nearest class. The arm is also above the table and is far larger than
    either object, so a greedy nearest-match would confidently mislabel it.
    """
    for b in scene.blobs:
        for name in ("cup", "die"):
            pr = priors.get(name)
            if not pr:
                continue
            if (
                pr["area_px"][0] <= b.area_px <= pr["area_px"][1]
                and pr["height_mm"][0] <= b.height_mm <= pr["height_mm"][1]
                and b.label == "unknown"
            ):
                b.label = name
    return scene


def success_state(scene: Scene) -> dict | None:
    """Build the state dict the canonical RLVR predicates consume.

    Returns ``None`` when either object is unlabelled — an unknown scene must produce NO
    verdict rather than a default one. A fabricated "failure" is worse than an absent
    label: it silently poisons the next training round.
    """
    cup, die = scene.by_label("cup"), scene.by_label("die")
    if cup is None or die is None:
        return None
    cx, cy, cz = cup.centroid_xyz
    r = 0.045  # cup footprint radius, metres — calibrate per cup
    return {
        "object_position": list(die.centroid_xyz),
        "bin_bounds": {
            "min": [cx - r, cy - r, min(cz, die.centroid_xyz[2]) - 0.02],
            "max": [cx + r, cy + r, max(cz, die.centroid_xyz[2]) + 0.05],
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_object_pose_depth.py")
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="report every blob so an operator can identify cup and die",
    )
    p.add_argument("--priors", type=Path, default=Path("outputs/object_priors.json"))
    p.add_argument("--warmup", type=int, default=45)
    p.add_argument(
        "--save-priors",
        action="store_true",
        help="with --calibrate, write priors around the two largest blobs",
    )
    args = p.parse_args(argv)

    depth, intr = capture_depth(args.warmup)
    scene = detect(depth, intr)
    print(
        f"[object-pose] table plane inliers: {scene.plane_inlier_frac * 100:.0f}% "
        f"({'usable' if scene.plane_inlier_frac > 0.25 else 'WEAK — verdicts unreliable'})"
    )
    print(f"[object-pose] blobs: {len(scene.blobs)}")

    if args.calibrate:
        for i, b in enumerate(scene.blobs):
            print(
                f"  [{i}] area={b.area_px:6d}px  px=({b.centroid_px[0]:5.0f},"
                f"{b.centroid_px[1]:5.0f})  height={b.height_mm:6.1f}mm  "
                f"xyz=({b.centroid_xyz[0]:+.3f},{b.centroid_xyz[1]:+.3f},"
                f"{b.centroid_xyz[2]:+.3f})"
            )
        print("\n  The ARM is also above the table and is usually the largest blob.")
        print(
            "  Identify cup and die by their height and position, then record priors."
        )
        return 0

    if not args.priors.exists():
        print(
            f"[object-pose] no priors at {args.priors} — run --calibrate first",
            file=sys.stderr,
        )
        return 2
    scene = classify(scene, json.loads(args.priors.read_text()))
    for b in scene.blobs:
        print(f"  {b.label:8s} area={b.area_px:6d}px height={b.height_mm:6.1f}mm")
    state = success_state(scene)
    if state is None:
        print(
            "[object-pose] VERDICT: UNKNOWN — cup and/or die not identified; "
            "no label emitted (an invented label poisons the next training round)"
        )
        return 3

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "src/lerobot-isaac-env/src")
    )
    from lerobot_isaac_env.outcome_verifier import object_in_box

    ok = bool(
        object_in_box(
            state["object_position"],
            state["bin_bounds"]["min"],
            state["bin_bounds"]["max"],
        )
    )
    print(
        f"[object-pose] VERDICT: {'SUCCESS' if ok else 'FAILURE'} "
        f"(die {'inside' if ok else 'outside'} the cup footprint)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
