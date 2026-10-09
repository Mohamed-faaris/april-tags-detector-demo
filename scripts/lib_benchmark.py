#!/usr/bin/env python3
"""Benchmark AprilTag detection libraries on frozen Blender renders.

Compares pupil-apriltags, apriltag (pipy), dt-apriltags and OpenCV ArUco on
the SAME frozen inputs (experiments/inputs/<id>/input.png + ground_truth.json)
with the SAME pose backend (cv2.solvePnP, true tag size) and the SAME metrics,
so differences reflect detection/corner quality. Also records mean
detection time per image.

Corner-order calibration: each library orders tag corners differently. On a
calibration experiment the script tries both object-point conventions x 4
cyclic shifts, picks the mapping with the lowest rotation error vs.
ground truth, and locks it for all experiments.

Outputs:
    experiments/outputs/lib_<name>/<exp_id>/{annotated.png, report.json}
    experiments/outputs/lib_<name>/summary.{csv,json}
    experiments/outputs/lib_compare.json   # head-to-head table

Run with uv (requires apriltag, dt-apriltags installed):
    uv run python scripts/lib_benchmark.py [--libs pupil dt apriltag opencv opencv_subpix]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_pose_sanity import (  # noqa: E402
    HEIGHT,
    MAX_POSITION_ERROR_M,
    MAX_RELATIVE_DISTANCE_ERROR_M,
    MAX_RELATIVE_TRANSLATION_ERROR_M,
    MAX_ROTATION_ERROR_DEG,
    REFERENCE_SIZE_M,
    TAG_SIZES_M,
    WIDTH,
    FOCAL_PIXELS,
    check_thresholds,
    expected_rotation_cv,
    load_poses_csv,
    rotation_error_deg,
    save_annotated_image,
)

ALL_LIBS = ("pupil", "dt", "apriltag", "opencv", "opencv_subpix")


def make_detector(lib: str, quad_decimate: float, quad_sigma: float,
                  refine_edges: int, decode_sharpening: float):
    if lib == "pupil":
        from pupil_apriltags import Detector
        return Detector(families="tag36h11", nthreads=2, quad_decimate=quad_decimate,
                        quad_sigma=quad_sigma, refine_edges=refine_edges,
                        decode_sharpening=decode_sharpening)
    if lib == "dt":
        import dt_apriltags
        return dt_apriltags.Detector(families="tag36h11", nthreads=2,
                                     quad_decimate=quad_decimate, quad_sigma=quad_sigma,
                                     refine_edges=refine_edges,
                                     decode_sharpening=decode_sharpening)
    if lib == "apriltag":
        import apriltag
        opts = apriltag.DetectorOptions(
            families="tag36h11", nthreads=4, quad_decimate=quad_decimate,
            quad_blur=quad_sigma, refine_edges=bool(refine_edges),
            refine_decode=False, refine_pose=False)
        return apriltag.Detector(opts)
    if lib in ("opencv", "opencv_subpix"):
        family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        if lib == "opencv_subpix":
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(family, params)
    raise ValueError(lib)


def detect_corners(detector, lib: str, gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Return [(tag_id, corners_4x2_px)] for one image."""
    if lib in ("pupil", "dt"):
        out = []
        for det in detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=(FOCAL_PIXELS, FOCAL_PIXELS, WIDTH / 2.0, HEIGHT / 2.0),
                tag_size=REFERENCE_SIZE_M):
            tag_id = int(det.tag_id)
            if tag_id in TAG_SIZES_M:
                out.append((tag_id, np.asarray(det.corners, dtype=float).reshape(4, 2)))
        return out
    if lib == "apriltag":
        return [(int(t.tag_id), np.asarray(t.corners, dtype=float).reshape(4, 2))
                for t in detector.detect(gray) if int(t.tag_id) in TAG_SIZES_M]
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [(int(tag_id), np.asarray(c, dtype=float).reshape(4, 2))
            for c, tag_id in zip(corners, ids.ravel()) if int(tag_id) in TAG_SIZES_M]


def obj_variants(size_m: float) -> list[np.ndarray]:
    h = size_m / 2.0
    return [
        np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], dtype=np.float64),
        np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64),
    ]


def cam_matrix() -> np.ndarray:
    return np.array([[FOCAL_PIXELS, 0, WIDTH / 2.0],
                     [0, FOCAL_PIXELS, HEIGHT / 2.0], [0, 0, 1]], dtype=np.float64)


def solvepnp_locked(corners: np.ndarray, size_m: float,
                    mapping: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    var_idx, shift = mapping[0], mapping[1]
    reverse = bool(mapping[2]) if len(mapping) > 2 else False
    obj = obj_variants(size_m)[var_idx]
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if reverse:
        pts = pts[::-1]
    ordered = np.ascontiguousarray(np.roll(pts, shift, axis=0))
    ok, rvec, tvec = cv2.solvePnP(obj, ordered, cam_matrix(), None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return tvec.reshape(3), cv2.Rodrigues(rvec)[0]


def calibrate_mapping(lib: str, detector, gray: np.ndarray,
                      specs) -> dict[int, tuple[int, int]]:
    """Pick the corner mapping per tag minimizing rotation error vs GT."""
    expected_rot = {tag_id: expected_rotation_cv(rot) for tag_id, _, _, rot in specs}
    mapping: dict[int, tuple[int, int]] = {}
    for tag_id, corners in detect_corners(detector, lib, gray):
        best: tuple[float, tuple[int, int, int]] | None = None
        # Libraries output the corners as a loop but may run clockwise or
        # counter-clockwise and start at any corner: search both windings x 4
        # shifts (the dihedral symmetries) x both object-point conventions.
        for var_idx, obj in enumerate(obj_variants(TAG_SIZES_M[tag_id])):
            for reverse in (False, True):
                pts = corners[::-1] if reverse else corners
                for shift in range(4):
                    ordered = np.ascontiguousarray(np.roll(pts, shift, axis=0))
                    ok, rvec, tvec = cv2.solvePnP(obj, ordered, cam_matrix(), None,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
                    if not ok:
                        continue
                    err = rotation_error_deg(cv2.Rodrigues(rvec)[0], expected_rot[tag_id])
                    if best is None or err < best[0]:
                        best = (err, (var_idx, shift, int(reverse)))
        if best is None:
            raise RuntimeError(f"[{lib}] solvePnP failed on calibration frame")
        if best[0] > 3.0:
            raise RuntimeError(f"[{lib}] tag {tag_id}: best mapping still "
                               f"{best[0]:.1f}deg off — check correspondence")
        mapping[tag_id] = best[1]
        print(f"[{lib}] tag {tag_id}: locked mapping variant={best[1][0]} "
              f"shift={best[1][1]} reversed={bool(best[1][2])} "
              f"(cal rot err {best[0]:.2f}deg)")
    if set(mapping) != set(TAG_SIZES_M):
        raise RuntimeError(f"[{lib}] calibration frame missing tags: {sorted(mapping)}")
    return mapping


def evaluate(exp_id: str, specs, corners_by_id: dict[int, np.ndarray],
             mapping: dict[int, tuple[int, int]], lib: str,
             config: dict) -> tuple[dict, dict, bool]:
    expected_pos = {tag_id: np.asarray(pos, dtype=float) for tag_id, _, pos, _ in specs}
    expected_rot = {tag_id: expected_rotation_cv(rot) for tag_id, _, _, rot in specs}
    detected: dict[int, dict] = {}
    estimated: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tag_id, corners in corners_by_id.items():
        pose = solvepnp_locked(corners, TAG_SIZES_M[tag_id], mapping[tag_id])
        if pose is None:
            continue
        t_est, r_est = pose
        r_err = rotation_error_deg(r_est, expected_rot[tag_id])
        detected[tag_id] = {
            "expected_camera_xyz_m": expected_pos[tag_id].tolist(),
            "estimated_camera_xyz_m": t_est.tolist(),
            "position_error_m": float(np.linalg.norm(t_est - expected_pos[tag_id])),
            "rotation_error_deg": r_err,
            "corners_px": np.asarray(corners).round(2).tolist(),
        }
        estimated[tag_id] = (t_est, r_est)
    # Reuse the shared threshold logic (needs the specs + estimated poses).
    fake_dets = {k: {"position_error_m": v["position_error_m"],
                     "rotation_error_deg": v["rotation_error_deg"]} for k, v in detected.items()}
    passed, relative = check_thresholds(fake_dets, estimated, specs)
    report = {
        "exp_id": exp_id, "library": lib, "detector_config": config,
        "pose_backend": "solvepnp_locked",
        "family": "tag36h11",
        "corner_mapping": {str(k): list(v) for k, v in mapping.items()},
        "detections": {str(k): v for k, v in detected.items()},
        "all_expected_ids_detected": set(detected) == set(TAG_SIZES_M),
        "thresholds": {
            "max_position_error_m": MAX_POSITION_ERROR_M,
            "max_relative_translation_error_m": MAX_RELATIVE_TRANSLATION_ERROR_M,
            "max_relative_distance_error_m": MAX_RELATIVE_DISTANCE_ERROR_M,
            "max_rotation_error_deg": MAX_ROTATION_ERROR_DEG,
        },
        "sanity_check_passed": bool(passed),
    }
    if relative is not None:
        report["relative_pose_from_id_0_to_id_4"] = relative
    return report, estimated, bool(passed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses-csv", type=Path, default=Path("experiments/poses_30.csv"))
    ap.add_argument("--inputs-dir", type=Path, default=Path("experiments/inputs"))
    ap.add_argument("--out-root", type=Path, default=Path("experiments/outputs"))
    ap.add_argument("--libs", nargs="+", default=list(ALL_LIBS), choices=ALL_LIBS)
    ap.add_argument("--quad-decimate", type=float, default=0.5)
    ap.add_argument("--quad-sigma", type=float, default=0.8)
    ap.add_argument("--refine-edges", type=int, default=1)
    ap.add_argument("--decode-sharpening", type=float, default=0.25)
    ap.add_argument("--calib-exp", default="001")
    args = ap.parse_args()

    rows = load_poses_csv(args.poses_csv)
    exp_ids = [str(r["exp_id"]).zfill(3) if str(r["exp_id"]).isdigit() else str(r["exp_id"])
               for r in rows]
    lib_table: dict[str, dict] = {}
    for lib in args.libs:
        print(f"===== library: {lib} =====")
        detector = make_detector(lib, args.quad_decimate, args.quad_sigma,
                                 args.refine_edges, args.decode_sharpening)
        config = {"library": lib, "quad_decimate": args.quad_decimate,
                  "quad_sigma": args.quad_sigma, "refine_edges": args.refine_edges,
                  "decode_sharpening": args.decode_sharpening,
                  "pose_backend": "solvepnp_locked"}
        out_dir = (args.out_root / f"lib_{lib}").resolve()
        # Calibrate corner mapping on frozen frames: scan experiments in order
        # and lock the first frame where both tags are detected and the best
        # mapping agrees with ground truth (<3 deg). Hard datasets may have
        # unusable frames, so don't insist on a single calibration exp.
        mapping: dict | None = None
        calib_used = ""
        for calib_exp in exp_ids:
            cal_gray = cv2.imread(str(args.inputs_dir / calib_exp / "input.png"),
                                  cv2.IMREAD_GRAYSCALE)
            if cal_gray is None:
                continue
            cal_gt = json.loads((args.inputs_dir / calib_exp / "ground_truth.json")
                                .read_text(encoding="utf-8"))
            cal_specs = [(int(t["id"]), float(t["size_m"]), tuple(t["expected_camera_xyz_m"]),
                          tuple(t["blender_euler_xyz_rad"])) for t in cal_gt["tags"]]
            try:
                mapping = calibrate_mapping(lib, detector, cal_gray, cal_specs)
                calib_used = calib_exp
                break
            except RuntimeError as exc:
                print(f"[{lib}] calibration on {calib_exp} unusable: {exc}")
        if mapping is None:
            raise RuntimeError(f"[{lib}] no calibratable frame in {len(exp_ids)} experiments")
        print(f"[{lib}] calibrated on experiment {calib_used}")
        # Warm-up (exclude Numba/JIT compile from timing for pupil/dt).
        detect_corners(detector, lib, cal_gray)

        summary_rows: list[dict] = []
        det_times_ms: list[float] = []
        all_passed = True
        for exp_id in exp_ids:
            gray = cv2.imread(str(args.inputs_dir / exp_id / "input.png"), cv2.IMREAD_GRAYSCALE)
            bgr = cv2.imread(str(args.inputs_dir / exp_id / "input.png"), cv2.IMREAD_COLOR)
            gt = json.loads((args.inputs_dir / exp_id / "ground_truth.json")
                            .read_text(encoding="utf-8"))
            specs = [(int(t["id"]), float(t["size_m"]), tuple(t["expected_camera_xyz_m"]),
                      tuple(t["blender_euler_xyz_rad"])) for t in gt["tags"]]
            t0 = time.perf_counter()
            dets = detect_corners(detector, lib, gray)
            det_times_ms.append((time.perf_counter() - t0) * 1000.0)
            corners_by_id = {tag_id: c for tag_id, c in dets if tag_id in TAG_SIZES_M}
            report, estimated, passed = evaluate(exp_id, specs, corners_by_id, mapping,
                                                 lib, config)
            exp_out = out_dir / exp_id
            exp_out.mkdir(parents=True, exist_ok=True)
            # Annotated overlay reuses the shared drawer via shim detections.
            class _Shim:
                def __init__(self, tag_id, corners, center):
                    self.tag_id = tag_id
                    self.corners = corners
                    self.center = center
            shims = {k: _Shim(k, np.asarray(v["corners_px"]),
                              np.mean(np.asarray(v["corners_px"]), axis=0))
                     for k, v in report["detections"].items()}
            det_rep = {k: report["detections"][str(k)] for k in shims}
            save_annotated_image(bgr, shims, estimated, det_rep, exp_out / "annotated.png")
            (exp_out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            rel = report.get("relative_pose_from_id_0_to_id_4", {})
            summary_rows.append({
                "exp_id": exp_id, "passed": passed,
                "detected_ids": ",".join(sorted(report["detections"])),
                "tag0_pos_err_m": report["detections"].get("0", {}).get("position_error_m", ""),
                "tag0_rot_err_deg": report["detections"].get("0", {}).get("rotation_error_deg", ""),
                "tag4_pos_err_m": report["detections"].get("4", {}).get("position_error_m", ""),
                "tag4_rot_err_deg": report["detections"].get("4", {}).get("rotation_error_deg", ""),
                "rel_trans_err_m": rel.get("translation_error_m", "") if isinstance(rel, dict) else "",
                "rel_dist_err_m": rel.get("distance_error_m", "") if isinstance(rel, dict) else "",
                "rel_rot_err_deg": rel.get("rotation_error_deg", "") if isinstance(rel, dict) else "",
                "detect_ms": round(det_times_ms[-1], 2),
            })
            all_passed = all_passed and passed
        with open(out_dir / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

        def mean(k):
            v = [r[k] for r in summary_rows if r[k] not in ("", None)]
            return sum(v) / len(v)
        stats = {
            "library": lib, "config": config, "num_experiments": len(summary_rows),
            "passed": sum(1 for r in summary_rows if r["passed"]),
            "mean_detect_ms": round(float(np.mean(det_times_ms)), 2),
            "mean_tag0_pos_m": mean("tag0_pos_err_m"),
            "mean_tag0_rot_deg": mean("tag0_rot_err_deg"),
            "mean_tag4_pos_m": mean("tag4_pos_err_m"),
            "mean_tag4_rot_deg": mean("tag4_rot_err_deg"),
            "mean_rel_trans_m": mean("rel_trans_err_m"),
            "mean_rel_dist_m": mean("rel_dist_err_m"),
            "mean_rel_rot_deg": mean("rel_rot_err_deg"),
        }
        (out_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
        lib_table[lib] = stats
        print(f"[{lib}] {stats['passed']}/{stats['num_experiments']} passed, "
              f"{stats['mean_detect_ms']:.1f} ms/image")

    compare_path = (args.out_root / "lib_compare.json").resolve()
    compare_path.write_text(json.dumps(lib_table, indent=2) + "\n")
    print(f"Wrote head-to-head table to {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
