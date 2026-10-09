#!/usr/bin/env python3
"""Calibrate a camera from checkerboard photos, saving viewer-ready NPZ.

1. Print a checkerboard (known square size), take 15-20 photos covering the
   whole frame: tilt/rotate the board, fill edges and corners.
2. Run:
    uv run python scripts/calibrate_camera.py --images 'calib/*.jpg' \\
        --pattern 9x6 --square-size 0.025 --out camera.npz
   --pattern is INNER corners (a 10x7-square board -> 9x6).
3. Use it:
    uv run april-tag-viewer --calibration camera.npz --tag-size 0.08

 Verifies by printing RMS reprojection error (good: < 0.5 px) and the matrix.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True,
                    help="glob for calibration photos, e.g. 'calib/*.jpg'")
    ap.add_argument("--pattern", default="9x6",
                    help="inner checkerboard corners COLSxROWS (default: 9x6)")
    ap.add_argument("--square-size", type=float, required=True,
                    help="checkerboard square edge length in metres")
    ap.add_argument("--out", type=Path, default=Path("camera.npz"))
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    pattern_size = (cols, rows)
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_size

    obj_points, img_points, shape = [], [], None
    files = sorted(glob.glob(args.images))
    if not files:
        raise SystemExit(f"No images match {args.images!r}")
    for path in files:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip unreadable {path}")
            continue
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not ok:
            print(f"no board in {path}")
            continue
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        obj_points.append(obj)
        img_points.append(corners)
        print(f"ok {path}")
    if len(obj_points) < 8:
        raise SystemExit(f"Only {len(obj_points)} usable views (need >= 8)")

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, shape, None, None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
             rms=np.asarray(rms), image_size=np.asarray(shape))
    print(f"\nRMS reprojection error: {rms:.3f} px ({'good' if rms < 0.5 else 'retake photos with more coverage'})")
    print(f"camera_matrix:\n{camera_matrix}")
    print(f"dist_coeffs: {dist_coeffs.ravel().tolist()}")
    print(f"saved to {args.out} -> use with: --calibration {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
