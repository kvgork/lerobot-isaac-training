#!/usr/bin/env python3
"""Framing gate — refuse a hardware session when the camera no longer matches training.

Track B prerequisite B0.1.

Session 1 of the bake-off ran with the camera 62 px / -98 px out and brightness 78
against a training value of 101, which made every image out-of-distribution. Four ACT
episodes were scored before anyone noticed, and the scoreboard voids them all:
"CAMERA REALIGNED ... all prior eps void". This gate exists to stop that first.

WHAT IT CHECKS, IN ORDER
------------------------
1. **Do the two images even show the same scene?**  A RANSAC similarity fit over the
   ORB matches; the INLIER RATIO is the test. Below ``--min-inlier-ratio`` the images do
   not relate by any rigid transform and no shift number means anything.
2. **Is it a pure pan?**  The fit also yields scale and rotation. A zoomed or rotated
   camera is not fixed by sliding it sideways, so those are reported separately.
3. **Translation** — only once 1 and 2 hold — against ``--max-shift-px``.
4. **Brightness** within ``--brightness-tol`` of the reference.

Exits non-zero on any failure, so it can gate a session script.

WHY STEP 1 EXISTS (learned the hard way, 2026-09-12)
----------------------------------------------------
The first version reported a median feature shift with no validity check. Against a live
rig it confidently and repeatedly reported "dx = -50 px", and that number did not move
when the operator physically repositioned the camera — because only **2% of matches were
geometrically consistent**. The median of 200 mostly-spurious matches is not a pan
measurement; it is a number-shaped artefact. An operator was sent to realign a camera on
the strength of it.

The lesson generalises past this script: a matcher that always returns matches will
always return a median, and a median always looks like a measurement. Validate that the
model you are fitting actually explains the data before you report a parameter from it.

The raw match COUNT is not that validation — it saturates at ``--max-matches`` and so
reads "200" for any pair of images whatsoever. It was misread as evidence once already.

MEASURED NOISE FLOOR
--------------------
Even on a genuinely matching scene, single-frame ORB has sd ~4.9 px and spread 15.2 px
on a static rig — half the 10 px bar. Default is 9 medianed captures. The sd is printed
every run and a warning fires when it is large relative to the bar. Treat a near-bar
result as "measure again with more samples", not as a pass.

CAPTURE PATH
------------
Captures via V4L2 by default — the same path robot-data-runner uses. A gate that
captures through pyrealsense while the runner reads V4L2 measures the wrong camera.
``--compare-paths`` grabs a RealSense frame too and reports the difference.

``--camera`` is REQUIRED. "Camera Index Instability" is a recorded gotcha for this rig,
there are eight /dev/video* nodes, and defaulting to one is how you measure a webcam.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATASET = "datasets/local/so101-pickplace-new"
DEFAULT_CAMERA_KEY = "observation.images.overhead"


# --------------------------------------------------------------------------- refs
def load_reference(dataset: Path, camera_key: str, n_frames: int):
    """Return (reference_frame_bgr, mean_brightness) from the training set."""
    import pyarrow.parquet as pq

    files = sorted(
        glob.glob(str(dataset / "data" / "**" / "*.parquet"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset / 'data'}")
    table = pq.read_table(files[0], columns=[camera_key])
    rows = min(n_frames, table.num_rows)
    if rows == 0:
        raise ValueError(f"{files[0]} has no rows")
    frames = []
    for i in range(rows):
        cell = table.column(camera_key)[i].as_py()
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode frame {i}")
        frames.append(img)
    return frames[0], float(np.mean([f.mean() for f in frames]))


def save_baseline(
    path: Path, frame, brightness: float, camera: str, samples: int
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / "reference.png"), frame)
    (path / "baseline.json").write_text(
        json.dumps(
            {
                "brightness": round(brightness, 3),
                "camera": camera,
                "samples": samples,
                "created": datetime.now().isoformat(timespec="seconds"),
                "frame_shape": list(frame.shape),
            },
            indent=2,
        )
        + "\n"
    )


def load_baseline(path: Path):
    img = cv2.imread(str(path / "reference.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"no reference.png under {path}")
    meta = json.loads((path / "baseline.json").read_text())
    return img, float(meta["brightness"]), meta


# ------------------------------------------------------------------------ capture
def grab_v4l2(device: str, warmup: int):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {device} via V4L2")
    try:
        frame = None
        for _ in range(max(1, warmup)):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(f"opened {device} but read no frame")
        return frame
    finally:
        cap.release()


def grab_realsense():
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    try:
        pipe.start()
        for _ in range(10):
            frames = pipe.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data())[:, :, ::-1].copy()
    finally:
        try:
            pipe.stop()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------------ measure
def fit_transform(ref, live, max_matches: int):
    """RANSAC similarity fit between two frames.

    Returns ``(dx, dy, scale, rot_deg, inlier_ratio, n_matches)``. The inlier ratio is
    the ONLY thing that says whether the rest is meaningful — see the module docstring.
    """
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=3000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0.0, 0.0, 1.0, 0.0, 0.0, 0

    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    if len(matches) < 10:
        return 0.0, 0.0, 1.0, 0.0, 0.0, len(matches)

    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None or inliers is None:
        return 0.0, 0.0, 1.0, 0.0, 0.0, len(matches)

    ratio = float(inliers.sum()) / len(matches)
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return float(M[0, 2]), float(M[1, 2]), scale, rot, ratio, len(matches)


def measure(ref, device: str, warmup: int, samples: int, max_matches: int):
    """Median fit over *samples* independent captures."""
    rows, brights = [], []
    for _ in range(max(1, samples)):
        live = grab_v4l2(device, warmup)
        if live.shape[:2] != ref.shape[:2]:
            live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
        rows.append(fit_transform(ref, live, max_matches))
        brights.append(float(live.mean()))
    a = np.array([r[:5] for r in rows], dtype=float)
    dx, dy, scale, rot, ratio = np.median(a, axis=0)
    sd = float(np.std(np.hypot(a[:, 0], a[:, 1])))
    return (
        float(dx),
        float(dy),
        float(scale),
        float(rot),
        float(ratio),
        float(np.median(brights)),
        sd,
        int(min(r[5] for r in rows)),
    )


def realign_hint(dx: float, dy: float, w: int, h: int) -> list[str]:
    """Translate a measured shift into a physical camera move.

    Sign convention established EMPIRICALLY against synthetic shifts, not reasoned
    about: a camera moved RIGHT makes the scene appear LEFT (dx < 0); a camera moved UP
    makes it appear LOWER (dy > 0, image y grows downward). The corrective move is in
    the same direction as the measured shift.
    """
    out: list[str] = []
    if abs(dx) >= 1.0:
        where, move = ("RIGHT", "LEFT") if dx < 0 else ("LEFT", "RIGHT")
        out.append(
            f"camera is too far {where} by ~{abs(dx):.0f} px "
            f"({abs(dx) / w * 100:.1f}% of frame width) -> move it {move}"
        )
    if abs(dy) >= 1.0:
        where, move = ("UP", "DOWN") if dy > 0 else ("DOWN", "UP")
        out.append(
            f"camera is too far {where} by ~{abs(dy):.0f} px "
            f"({abs(dy) / h * 100:.1f}% of frame height) -> move it {move}"
        )
    if not out:
        out.append("translation is within a pixel")
    return out


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="_framing_gate.py", description=__doc__.split("\n")[0]
    )
    p.add_argument(
        "--camera", required=True, help="V4L2 device. REQUIRED — do not guess."
    )
    p.add_argument("--reference-dataset", default=DEFAULT_DATASET, type=Path)
    p.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="measure against a saved baseline instead of the training set",
    )
    p.add_argument(
        "--save-baseline",
        type=Path,
        default=None,
        help="capture the CURRENT rig as a new baseline, then exit",
    )
    p.add_argument("--max-shift-px", type=float, default=10.0)
    p.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.25,
        help="below this the images are not the same scene and no shift is "
        "reportable (default 0.25)",
    )
    p.add_argument(
        "--max-scale-dev", type=float, default=0.05, help="allowed |scale-1|"
    )
    p.add_argument("--max-rot-deg", type=float, default=2.0)
    p.add_argument("--brightness-ref", type=float, default=None)
    p.add_argument("--brightness-tol", type=float, default=0.10)
    p.add_argument("--ref-frames", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--max-matches", type=int, default=400)
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--compare-paths", action="store_true")
    args = p.parse_args(argv)

    if args.save_baseline is not None:
        try:
            frames = [
                grab_v4l2(args.camera, args.warmup) for _ in range(max(1, args.samples))
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"[framing-gate] FATAL: capture failed: {exc}", file=sys.stderr)
            return 2
        bright = float(np.median([f.mean() for f in frames]))
        save_baseline(args.save_baseline, frames[-1], bright, args.camera, args.samples)
        print(f"[framing-gate] baseline written: {args.save_baseline}")
        print(f"[framing-gate]   camera={args.camera} brightness={bright:.1f}")
        print("[framing-gate]")
        print(
            "[framing-gate] WARNING - a baseline defines what 'correct framing' MEANS."
        )
        print(
            "[framing-gate]   Every policy already trained saw the OLD framing. Measuring"
        )
        print(
            "[framing-gate]   against this new one reports PASS while those policies still"
        )
        print(
            "[framing-gate]   see out-of-distribution images. Adopt one only when you are"
        )
        print(
            "[framing-gate]   about to RETRAIN against it, or have deliberately repositioned"
        )
        print(
            "[framing-gate]   and accept that prior checkpoints are no longer comparable."
        )
        print(f"[framing-gate]   Use it with: --baseline {args.save_baseline}")
        return 0

    try:
        if args.baseline is not None:
            ref, ref_bright, meta = load_baseline(args.baseline)
            print(
                f"[framing-gate] baseline   : {args.baseline} "
                f"(captured {meta.get('created', '?')} from {meta.get('camera', '?')})"
            )
            print(
                "[framing-gate]   NOTE: a PASS against a saved baseline does NOT prove the "
                "rig matches what the policy saw."
            )
        else:
            ref, ref_bright = load_reference(
                args.reference_dataset, args.camera_key, args.ref_frames
            )
            print(
                f"[framing-gate] reference  : {args.reference_dataset} :: {args.camera_key}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[framing-gate] FATAL: could not load reference: {exc}", file=sys.stderr)
        return 2
    if args.brightness_ref is not None:
        ref_bright = args.brightness_ref

    try:
        dx, dy, scale, rot, ratio, live_bright, sd, n = measure(
            ref, args.camera, args.warmup, args.samples, args.max_matches
        )
        live = grab_v4l2(args.camera, args.warmup)
        if live.shape[:2] != ref.shape[:2]:
            live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
    except Exception as exc:  # noqa: BLE001
        print(f"[framing-gate] FATAL: capture failed: {exc}", file=sys.stderr)
        return 2

    shift = float(np.hypot(dx, dy))
    bright_delta = (
        abs(live_bright - ref_bright) / ref_bright if ref_bright else float("inf")
    )
    same_scene = ratio >= args.min_inlier_ratio

    print(f"[framing-gate] camera     : {args.camera} (V4L2 — the runner's own path)")
    print(
        f"[framing-gate] scene match: {ratio * 100:.0f}% of {n} matches are geometrically "
        f"consistent (bar >= {args.min_inlier_ratio * 100:.0f}%)"
    )

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_dir / "reference.png"), ref)
        cv2.imwrite(str(args.save_dir / "live.png"), live)
        cv2.imwrite(
            str(args.save_dir / "blend.png"), cv2.addWeighted(ref, 0.5, live, 0.5, 0)
        )
        print(
            f"[framing-gate] saved      : {args.save_dir}/{{reference,live,blend}}.png"
        )

    if not same_scene:
        print(
            "[framing-gate] FAIL: the live view and the reference are NOT the same scene.",
            file=sys.stderr,
        )
        print(
            f"  Only {ratio * 100:.0f}% of matches fit any rigid transform, so no shift is",
            file=sys.stderr,
        )
        print(
            "  reportable and no realignment advice would be meaningful.",
            file=sys.stderr,
        )
        print(
            "  Likely causes: wrong camera index, camera pointed elsewhere, or the physical",
            file=sys.stderr,
        )
        print(
            "  scene has changed since training (different objects, surface, or layout).",
            file=sys.stderr,
        )
        print("  Look at the saved frames before moving anything.", file=sys.stderr)
        return 1

    print(
        f"[framing-gate] shift      : dx={dx:+.1f} dy={dy:+.1f} |{shift:.1f}| px "
        f"(bar < {args.max_shift_px}, sd {sd:.1f} over {args.samples} captures)"
    )
    print(f"[framing-gate] scale/rot  : scale={scale:.3f} rot={rot:+.2f} deg")
    print(
        f"[framing-gate] brightness : live={live_bright:.1f} ref={ref_bright:.1f} "
        f"delta={bright_delta * 100:.1f}% (bar <= {args.brightness_tol * 100:.0f}%)"
    )
    if sd > args.max_shift_px / 3.0:
        print(
            f"[framing-gate] WARNING: sd {sd:.1f} px is large relative to the "
            f"{args.max_shift_px} px bar — raise --samples before trusting a near-bar result"
        )

    failures = []
    if abs(scale - 1.0) > args.max_scale_dev:
        failures.append(
            f"scale differs by {abs(scale - 1) * 100:.1f}% — the camera is at a different "
            f"distance or zoom; sliding it sideways will not fix this"
        )
    if abs(rot) > args.max_rot_deg:
        failures.append(f"camera is rotated {rot:+.1f} deg — level it before panning")
    if shift >= args.max_shift_px:
        failures.append(f"framing shift {shift:.1f} px >= {args.max_shift_px} px")
    if bright_delta > args.brightness_tol:
        failures.append(
            f"brightness off by {bright_delta * 100:.1f}% (live {live_bright:.1f} vs "
            f"ref {ref_bright:.1f})"
        )

    print("[framing-gate] realign    :")
    for line in realign_hint(dx, dy, ref.shape[1], ref.shape[0]):
        print(f"[framing-gate]   - {line}")
    if bright_delta > args.brightness_tol:
        direction = "brighter" if live_bright < ref_bright else "darker"
        print(
            f"[framing-gate]   - lighting: make the scene {direction} "
            f"(live {live_bright:.1f} -> target {ref_bright:.1f})"
        )

    if failures:
        print("[framing-gate] FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("[framing-gate] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
