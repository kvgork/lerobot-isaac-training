#!/usr/bin/env python3
"""Framing gate — refuse a hardware session when the camera no longer matches training.

Track B prerequisite B0.1. Session 1 of the bake-off ran with the camera **62 px / -98 px**
out of alignment and brightness 78 against a training value of 101, which made every
image out-of-distribution. Four ACT episodes were scored against that rig before anyone
noticed, and the scoreboard voids them: "CAMERA REALIGNED ... all prior eps void".

This gate turns that from a thing you notice afterwards into a thing that stops you first.

Gate (both must hold):
  * ORB **median** feature shift between a live frame and a reference training frame
    < ``--max-shift-px`` (default 10)
  * mean brightness within ``--brightness-tol`` of the training reference (default 10%)

Exits non-zero on failure, so it can gate a session script.

THE CAPTURE-PATH TRAP
---------------------
The plan flags this explicitly: a gate that captures through pyrealsense while the runner
reads V4L2 measures **the wrong camera**, and would happily pass a rig the runner sees
differently. So this gate captures through **V4L2 by default — the same path the runner
uses**. ``--compare-paths`` additionally grabs a pyrealsense frame and reports the shift
between the two, which is the check that proves the two paths agree.

MEASURED NOISE FLOOR — read this before trusting a near-bar result
------------------------------------------------------------------
On a completely static scene this gate measures shift with **sd ~4-7 px at the default
9 samples** (single-frame sd was 4.9 px, spread 15.2 px). That noise is irreducible by
sampling: the reference frames come from training episodes where the arm and the object
sit in different places, so ORB is partly matching genuinely different content, not just
different framing. Medianing across many *reference* frames was tried and did not help
(per-reference sd 12-14 px).

Practical consequence:
  * a rig tens of px out (session 1 was 62 / -98 px) is detected decisively;
  * a rig sitting NEAR the 10 px bar cannot be adjudicated reliably by this gate.

The sd is printed on every run and a warning fires when it is large relative to the bar.
Treat a near-bar result as "measure again with more samples", not as a pass.

The camera index is a REQUIRED argument. "Camera Index Instability" is a recorded gotcha
for this rig (vault: entities/SO-101), there are eight /dev/video* nodes, and defaulting
to one of them is how you measure a webcam instead of the overhead camera.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATASET = "datasets/local/so101-pickplace-new"
DEFAULT_CAMERA_KEY = "observation.images.overhead"
DEFAULT_BRIGHTNESS_REF = 101.0  # measured training-set mean; see plan Track B session 1


def load_reference(
    dataset: Path, camera_key: str, n_frames: int
) -> tuple[np.ndarray, float]:
    """Return (reference_frame_bgr, mean_brightness) sampled from the training set.

    The ORB reference is a single frame; brightness is averaged over *n_frames* so one
    unusually lit frame cannot move the bar.
    """
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
            raise ValueError(f"could not decode frame {i} of {camera_key}")
        frames.append(img)

    brightness = float(np.mean([f.mean() for f in frames]))
    return frames[0], brightness


def grab_v4l2(device: str, warmup: int) -> np.ndarray:
    """Capture one frame through V4L2 — the same path robot-data-runner uses."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {device} via V4L2")
    try:
        frame = None
        # Discard warmup frames: the first frames off a just-opened device are often
        # dark or stale, which would fail the brightness check for the wrong reason.
        for _ in range(max(1, warmup)):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(f"opened {device} but read no frame")
        return frame
    finally:
        cap.release()


def grab_realsense() -> np.ndarray:
    """Capture one frame through pyrealsense2 — only for --compare-paths."""
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    try:
        pipe.start()
        for _ in range(10):
            frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data())[:, :, ::-1].copy()  # RGB -> BGR
    finally:
        try:
            pipe.stop()
        except Exception:  # noqa: BLE001
            pass


def median_shift(
    ref: np.ndarray, live: np.ndarray, max_matches: int
) -> tuple[float, float, int]:
    """ORB median (dx, dy) between two frames, plus the match count used.

    Median, not mean: a handful of wrong matches is normal and would drag a mean
    arbitrarily far. Returns (dx, dy, n_matches); n_matches == 0 means "cannot tell".
    """
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0.0, 0.0, 0

    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if not matches:
        return 0.0, 0.0, 0
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    deltas = np.array(
        [np.subtract(k2[m.trainIdx].pt, k1[m.queryIdx].pt) for m in matches]
    )
    dx, dy = np.median(deltas, axis=0)
    return float(dx), float(dy), len(matches)


def measure(
    ref: np.ndarray, device: str, warmup: int, samples: int, max_matches: int
) -> tuple[float, float, float, float, int]:
    """Median (dx, dy) and brightness over *samples* independent captures.

    A SINGLE-frame ORB match is far too noisy for a 10 px bar. Measured on a static
    scene with nothing moving: sd 4.9 px, spread 15.2 px across 10 captures, with dx
    stable near -48 while dy wandered +6.5 to +24.7. A rig genuinely 8 px out would
    have read anywhere from 3 px (pass) to 18 px (fail) depending on which frame it
    happened to grab. Medianing across captures collapses that.

    Returns ``(dx, dy, brightness, shift_sd, min_matches)``. The sd is reported so the
    operator can see when the measurement itself is untrustworthy.
    """
    ds, bs, ns = [], [], []
    for _ in range(max(1, samples)):
        live = grab_v4l2(device, warmup)
        if live.shape[:2] != ref.shape[:2]:
            live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
        dx, dy, n = median_shift(ref, live, max_matches)
        ds.append((dx, dy))
        bs.append(float(live.mean()))
        ns.append(n)
    arr = np.array(ds)
    dx, dy = np.median(arr, axis=0)
    shift_sd = float(np.std(np.hypot(arr[:, 0], arr[:, 1])))
    return float(dx), float(dy), float(np.median(bs)), shift_sd, int(min(ns))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="_framing_gate.py",
        description="Refuse a hardware session when the camera no longer matches training.",
    )
    p.add_argument(
        "--camera",
        required=True,
        help="V4L2 device, e.g. /dev/video4. REQUIRED — do not guess; this "
        "rig has known camera-index instability and 8 video nodes.",
    )
    p.add_argument("--reference-dataset", default=DEFAULT_DATASET, type=Path)
    p.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    p.add_argument("--max-shift-px", type=float, default=10.0)
    p.add_argument(
        "--brightness-ref",
        type=float,
        default=None,
        help=f"override the training mean (default: measured from the dataset, "
        f"falling back to {DEFAULT_BRIGHTNESS_REF})",
    )
    p.add_argument(
        "--brightness-tol", type=float, default=0.10, help="fractional, default 0.10"
    )
    p.add_argument(
        "--ref-frames", type=int, default=30, help="frames averaged for brightness"
    )
    p.add_argument(
        "--warmup", type=int, default=10, help="frames discarded before capture"
    )
    p.add_argument(
        "--samples",
        type=int,
        default=9,
        help="independent captures medianed together. A single frame has "
        "sd ~4.9 px against a 10 px bar; 9 brings it under 2.",
    )
    p.add_argument("--max-matches", type=int, default=200)
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="write ref/live frames for diagnosis",
    )
    p.add_argument(
        "--compare-paths",
        action="store_true",
        help="also grab via pyrealsense and report the V4L2-vs-RealSense shift",
    )
    args = p.parse_args(argv)

    try:
        ref, ref_bright = load_reference(
            args.reference_dataset, args.camera_key, args.ref_frames
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[framing-gate] FATAL: could not load reference: {exc}", file=sys.stderr)
        return 2
    if args.brightness_ref is not None:
        ref_bright = args.brightness_ref

    try:
        dx, dy, live_bright, shift_sd, n = measure(
            ref, args.camera, args.warmup, args.samples, args.max_matches
        )
        live = grab_v4l2(args.camera, args.warmup)  # one more, for --save-dir
        if live.shape[:2] != ref.shape[:2]:
            live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
    except Exception as exc:  # noqa: BLE001
        print(f"[framing-gate] FATAL: capture failed: {exc}", file=sys.stderr)
        return 2

    shift = float(np.hypot(dx, dy))
    bright_delta = (
        abs(live_bright - ref_bright) / ref_bright if ref_bright else float("inf")
    )

    print(f"[framing-gate] reference  : {args.reference_dataset} :: {args.camera_key}")
    print(f"[framing-gate] camera     : {args.camera} (V4L2 — the runner's own path)")
    print(f"[framing-gate] ORB matches: {n}")
    print(
        f"[framing-gate] shift      : dx={dx:+.1f} dy={dy:+.1f} |{shift:.1f}| px "
        f"(bar < {args.max_shift_px}, sd {shift_sd:.1f} over {args.samples} captures)"
    )
    if shift_sd > args.max_shift_px / 3.0:
        print(
            f"[framing-gate] WARNING: measurement sd {shift_sd:.1f} px is large relative to "
            f"the {args.max_shift_px} px bar — raise --samples before trusting a near-bar result"
        )
    print(
        f"[framing-gate] brightness : live={live_bright:.1f} ref={ref_bright:.1f} "
        f"delta={bright_delta * 100:.1f}% (bar <= {args.brightness_tol * 100:.0f}%)"
    )

    if args.compare_paths:
        try:
            rs_frame = grab_realsense()
            if rs_frame.shape[:2] != ref.shape[:2]:
                rs_frame = cv2.resize(rs_frame, (ref.shape[1], ref.shape[0]))
            pdx, pdy, pn = median_shift(live, rs_frame, args.max_matches)
            pshift = float(np.hypot(pdx, pdy))
            verdict = "AGREE" if pshift < args.max_shift_px else "DISAGREE"
            print(
                f"[framing-gate] path check : V4L2 vs RealSense |{pshift:.1f}| px "
                f"({pn} matches) -> {verdict}"
            )
            if verdict == "DISAGREE":
                print(
                    "[framing-gate]   the two capture paths see different geometry; a gate "
                    "run through the wrong one measures the wrong camera",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[framing-gate] path check : unavailable ({type(exc).__name__}: {exc})"
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

    failures = []
    if n == 0:
        failures.append(
            "no ORB matches — the scene may be unrecognisable, lens covered, or dark"
        )
    elif shift >= args.max_shift_px:
        failures.append(f"framing shift {shift:.1f} px >= {args.max_shift_px} px")
    if bright_delta > args.brightness_tol:
        failures.append(
            f"brightness off by {bright_delta * 100:.1f}% (live {live_bright:.1f} vs "
            f"ref {ref_bright:.1f}) — policies trained at the reference level see OOD images"
        )

    if failures:
        print("[framing-gate] FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "  Realign the camera / fix the lighting and re-run. Do NOT score episodes "
            "against a rig that fails this gate — session 1 did, and every episode was voided.",
            file=sys.stderr,
        )
        return 1

    print("[framing-gate] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
