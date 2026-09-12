#!/usr/bin/env python3
"""Object pose for the SO-101 workspace — the RLVR pose source.

WHY THIS EXISTS
---------------
``binary_success_verifier`` and ``lerobot_isaac_env.outcome_verifier`` have had sound
success predicates for months (``object_in_box``, ``object_lifted``, ...), all consuming
a plain state dict. Nothing on this rig ever produced that dict, so every predicate was
unreachable on hardware and every episode needed an operator keypress. That gap blocks
automated eval scoring AND the self-improving collection loop, where an unlabelled round
trains on its own failures.

DESIGN — each object is found by the signal that actually separates it
---------------------------------------------------------------------
Depth alone does not work here, and the reasons are specific to this rig:

* the DIE is a 16 mm cube. Depth cannot resolve it against the table, and it was
  invisible to blob detection at every threshold tried. It IS bright red, and nothing
  else red sits in the workspace, so it is found by colour.
* the CUP is white with green leaves. Depth sees it as a raised structure but so is the
  ARM, which is also white — size and height priors cannot separate them reliably
  because the arm's apparent size changes with every pose. The green leaves are present
  on the cup and absent on the arm at ANY arm pose, so green is the cup's identity.
  Measured: a tight green range returns 5 blobs, all within 41 px of the cup, and
  nothing elsewhere in the frame.
* the ARM is white and raised with no green — explicitly rejected rather than
  hopefully-not-matched.

The die's 3-D position comes from intersecting its pixel ray with the fitted TABLE
PLANE, not from the depth value at its pixel. Depth on a 16 mm object is unreliable
(a probe read it 14 cm off the table), whereas the plane is fitted from ~50 000 points.

MEASURED LIMITS
---------------
* Table-plane fit is 48-52% of the workspace band. Adequate for "is the die in the cup",
  not for millimetre pose.
* Cup detection needs the green leaves visible; if the arm occludes them the cup is
  reported missing rather than guessed at.
* No ground-truth validation yet. ``--calibrate`` prints everything it sees so an
  operator can check it before any verdict is trusted.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WORKSPACE_NEAR_M, WORKSPACE_FAR_M = 0.35, 0.85
PLANE_TOL_M = 0.005
OBJECT_MIN_H_M, OBJECT_MAX_H_M = 0.010, 0.200

#: Red die. Red wraps the hue circle, hence two bands.
DIE_HSV = (((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255)))
DIE_AREA_PX = (120, 800)
DIE_ASPECT = (0.70, 1.40)

#: Green leaves on the cup. Deliberately TIGHT: a wider range picks up other green in
#: the room, while this range returned only cup pixels on the calibration frame.
CUP_GREEN_HSV = ((40, 80, 50), (80, 255, 255))
CUP_GREEN_MIN_PX = 30


@dataclass
class Detection:
    label: str
    centroid_px: tuple[float, float]
    centroid_xyz: tuple[float, float, float]
    area_px: int
    extra: dict


def capture(warmup: int = 60):
    """One depth+colour frame, depth ALIGNED to colour so pixels correspond."""
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(max(1, warmup)):
            frames = pipe.wait_for_frames()
        frames = align.process(frames)
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        intr = (
            prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        )
    finally:
        pipe.stop()
    return color, depth * scale, intr


def fit_table_plane(depth_m, intr, seed: int = 0):
    """RANSAC the table. Returns (normal, d, inlier_fraction), normal pointing up."""
    band = (depth_m > WORKSPACE_NEAR_M) & (depth_m < WORKSPACE_FAR_M)
    if band.sum() < 5000:
        raise ValueError(f"only {int(band.sum())} px in the workspace depth band")
    ys, xs = np.nonzero(band)
    Z = depth_m[band]
    X = (xs - intr.ppx) / intr.fx * Z
    Y = (ys - intr.ppy) / intr.fy * Z
    pts = np.stack([X, Y, Z], axis=1)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(300):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ p[0]
        inl = int((np.abs(pts @ n + d) < PLANE_TOL_M).sum())
        if best is None or inl > best[0]:
            best = (inl, n, d)
    inl, n, d = best
    if np.median(-(pts @ n + d)) < 0:
        n, d = -n, -d
    return n, d, inl / len(pts), (pts, ys, xs)


def ray_plane(px, py, intr, n, d):
    """Where the ray through pixel (px,py) meets the table plane.

    Used for the DIE instead of its depth pixel: depth on a 16 mm object is unreliable
    (a probe put it 14 cm off the table), while the plane comes from ~50 000 points.
    """
    ray = np.array([(px - intr.ppx) / intr.fx, (py - intr.ppy) / intr.fy, 1.0])
    denom = n @ ray
    if abs(denom) < 1e-9:
        return None
    return tuple(float(v) for v in ray * (-d / denom))


def _mask(hsv, bands):
    m = None
    for lo, hi in bands:
        b = cv2.inRange(hsv, lo, hi)
        m = b if m is None else cv2.bitwise_or(m, b)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_die(color, intr, n, d) -> Detection | None:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(_mask(hsv, DIE_HSV), 8)
    best = None
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (DIE_AREA_PX[0] <= a <= DIE_AREA_PX[1]):
            continue
        if not (DIE_ASPECT[0] <= w / max(h, 1) <= DIE_ASPECT[1]):
            continue
        # The die is ON THE TABLE, so its ray must meet the plane inside the workspace
        # band. A red blob at the frame edge extrapolates the plane far away (measured:
        # z=1.258 m for a red object at x=21) and is not a candidate at all.
        probe = ray_plane(float(cents[i][0]), float(cents[i][1]), intr, n, d)
        if probe is None or not (WORKSPACE_NEAR_M <= probe[2] <= WORKSPACE_FAR_M):
            continue
        # Pick the most SQUARE candidate, not the largest. A cube viewed from above is
        # square; "largest red blob in the workspace" picked a 33x22 distractor over the
        # 24x21 die. Squareness is a property of the object, not of this one frame.
        score = abs(w / max(h, 1) - 1.0)
        if best is None or score < best[0]:
            best = (score, a, float(cents[i][0]), float(cents[i][1]), w, h)
    if best is None:
        return None
    _score, a, cx, cy, w, h = best
    xyz = ray_plane(cx, cy, intr, n, d)
    if xyz is None:
        return None
    return Detection("die", (cx, cy), xyz, a, {"w": w, "h": h})


def find_cup(color, depth_m, intr, n, d, cloud) -> Detection | None:
    """A raised depth region that CONTAINS green pixels. The arm is raised too, but white."""
    pts, ys, xs = cloud
    h = -(pts @ n + d)
    above = (h > OBJECT_MIN_H_M) & (h < OBJECT_MAX_H_M)
    raised = np.zeros(depth_m.shape, np.uint8)
    raised[ys[above], xs[above]] = 255
    raised = cv2.morphologyEx(raised, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    green = _mask(hsv, [CUP_GREEN_HSV])
    if int(green.sum() // 255) < CUP_GREEN_MIN_PX:
        return None

    # Dilate the green so nearby raised components belonging to one cup are joined —
    # watershed over-split this cup into two blobs on the calibration frame.
    seed = cv2.dilate(green, np.ones((25, 25), np.uint8))
    cup_mask = cv2.bitwise_and(raised, seed)
    if int(cup_mask.sum() // 255) < 50:
        return None
    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(cup_mask, 8)
    i = int(np.argmax([stats[j, cv2.CC_STAT_AREA] for j in range(1, nlab)]) + 1)
    a = int(stats[i, cv2.CC_STAT_AREA])
    cx, cy = float(cents[i][0]), float(cents[i][1])
    sel = lab[ys[above], xs[above]] == i
    xyz = (
        tuple(float(v) for v in pts[above][sel].mean(axis=0))
        if sel.sum()
        else ray_plane(cx, cy, intr, n, d)
    )
    radius_px = float(np.sqrt(a / np.pi))
    z = xyz[2] if xyz else 0.6
    radius_m = radius_px / intr.fx * z
    return Detection(
        "cup",
        (cx, cy),
        xyz,
        a,
        {
            "radius_m": radius_m,
            "height_mm": float(np.median(h[above][sel]) * 1000) if sel.sum() else 0.0,
        },
    )


def success_state(die: Detection | None, cup: Detection | None) -> dict | None:
    """State dict for the canonical predicates, or None when the scene is unreadable.

    Returning None rather than a default is deliberate: an invented "failure" silently
    poisons the next training round, and the flywheel cannot detect that on its own.
    """
    if die is None or cup is None:
        return None
    cx, cy, cz = cup.centroid_xyz
    r = cup.extra["radius_m"]
    return {
        "object_position": list(die.centroid_xyz),
        "bin_bounds": {
            "min": [cx - r, cy - r, cz - 0.10],
            "max": [cx + r, cy + r, cz + 0.10],
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="_object_pose_depth.py")
    p.add_argument("--calibrate", action="store_true", help="print everything seen")
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--save", type=Path, default=None, help="write an annotated frame")
    args = p.parse_args(argv)

    color, depth_m, intr = capture(args.warmup)
    n, d, frac, cloud = fit_table_plane(depth_m, intr)
    print(
        f"[object-pose] table plane {frac * 100:.0f}% inliers "
        f"({'usable' if frac > 0.25 else 'WEAK'})"
    )

    die = find_die(color, intr, n, d)
    cup = find_cup(color, depth_m, intr, n, d, cloud)
    for name, det in (("die", die), ("cup", cup)):
        if det is None:
            print(f"[object-pose] {name}: NOT FOUND")
        else:
            print(
                f"[object-pose] {name}: px=({det.centroid_px[0]:.0f},{det.centroid_px[1]:.0f}) "
                f"area={det.area_px}px xyz=({det.centroid_xyz[0]:+.3f},"
                f"{det.centroid_xyz[1]:+.3f},{det.centroid_xyz[2]:+.3f}) {det.extra}"
            )

    if args.save:
        out = color.copy()
        if cup:
            cv2.circle(
                out, tuple(int(v) for v in cup.centroid_px), 34, (0, 255, 255), 2
            )
            cv2.putText(
                out,
                "CUP",
                (int(cup.centroid_px[0]) - 20, int(cup.centroid_px[1]) - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        if die:
            cv2.rectangle(
                out,
                (int(die.centroid_px[0]) - 18, int(die.centroid_px[1]) - 18),
                (int(die.centroid_px[0]) + 18, int(die.centroid_px[1]) + 18),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                out,
                "DIE",
                (int(die.centroid_px[0]) + 22, int(die.centroid_px[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), out)
        print(f"[object-pose] wrote {args.save}")

    state = success_state(die, cup)
    if state is None:
        print("[object-pose] VERDICT: UNKNOWN — no label emitted")
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
        f"(die {'inside' if ok else 'outside'} the cup)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
