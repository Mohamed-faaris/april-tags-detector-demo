"""Live AprilTag pose viewer with the lowest visible ID as the origin."""

from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

# Best benchmarked detector settings (30/30 on easy set, best wide-set
# accuracy where detected): quad_decimate=0.5, quad_sigma=0.8.
BEST_QUAD_DECIMATE = 0.5
BEST_QUAD_SIGMA = 0.8

# Per-ID printed tag edge lengths in metres for this setup. Command-line
# --marker-size entries override these values.
# MARKER_SIZES_M: dict[int, float] = {
#     0: 0.095,
#     1: 0.095,
#     2: 0.095,
#     3: 0.095,
#     4: 0.06725,
#     5: 0.06725,
#     6: 0.06725,
#     7: 0.06725,
#     8: 0.06725,
#     9: 0.06725,
#     10: 0.06725,
#     11: 0.06725,
# }
MARKER_SIZES_M: dict[int, float] = {
    0: 0.0885,
    1: 0.0885,
    2: 0.0885,
    3: 0.0885,
    4: 0.05750,
    5: 0.05750,
    6: 0.05750,
    7: 0.05750,
    8: 0.05750,
    9: 0.05750,
    10: 0.05750,
    11: 0.05750,
}


@dataclass
class Overlays:
    box_boundary: bool = True
    top_right_indicator: bool = True
    pose: bool = True
    axes: bool = True
    help: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="0", help="camera index, video file, or image file (default: 1)")
    parser.add_argument("--tag-size", type=float, default=0.05, help="printed tag edge length in metres (default: 0.05)")
    parser.add_argument("--marker-size", action="append", default=[], metavar="ID=METRES",
                        help="per-ID edge length override; repeat for multiple IDs, e.g. --marker-size 3=0.08")
    parser.add_argument("--dictionary", default="36h11", choices=("16h5", "25h9", "36h11"))
    parser.add_argument("--calibration", type=Path,
                        help="existing calibration file from tools/calibrator "
                             "(camera_calibration.npz, camera_calibration.json, or camera_info.yaml)")
    parser.add_argument("--fov", type=float, default=60.0, help="approximate horizontal camera FOV in degrees without calibration")
    parser.add_argument("--axis-length", type=float, default=0.03, help="drawn axis length in metres")
    parser.add_argument("--quad-decimate", type=float, default=BEST_QUAD_DECIMATE,
                        help="detector decimation (best benchmarked: %(default)s)")
    parser.add_argument("--quad-sigma", type=float, default=BEST_QUAD_SIGMA,
                        help="detector blur sigma (best benchmarked: %(default)s)")
    parser.add_argument("--pose-backend", default="solvepnp", choices=("solvepnp", "tag_pose"),
                        help="solvepnp: cv2.solvePnP with true tag size (best benchmarked); "
                             "tag_pose: detector pose scaled to tag size")
    parser.add_argument("--smooth-alpha", type=float, default=0.45,
                        help="pose smoothing 0..1 (higher = snappier, lower = steadier; 1 disables)")
    parser.add_argument("--max-reproj-px", type=float, default=5.0,
                        help="reject detections whose solvePnP reprojection error exceeds this (px); "
                             "raise if valid tags show '?pose' (e.g. uncalibrated lens)")
    parser.add_argument("--max-missed", type=int, default=5,
                        help="drop smoothed tracks unseen for this many frames")
    return parser.parse_args()


@contextmanager
def suppress_native_noise():
    """Silence C-level printf chatter (e.g. AprilTag union-find 'minima'
    diagnostics) emitted by the detector around each detect() call."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def load_calibration_file(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None, float | None]:
    """Load camera calibration from the econ calibrator's artifacts (untouched).

    Accepts any of its machine-readable outputs:
      camera_calibration.npz  (camera_matrix, dist_coeffs[, image_size, rms*])
      camera_calibration.json (camera_matrix.{fx,fy,cx,cy|matrix_3x3}, distortion_coefficients)
      camera_info.yaml        (ROS camera_info: camera_matrix.data, distortion_coefficients.data)
    Returns (camera_matrix 3x3, dist_coeffs, calib image (w,h) or None, rms or None).
    """
    import json

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            keys = set(data.files)
            camera = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
            dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
            size = None
            if "image_size" in keys and np.asarray(data["image_size"]).size == 2:
                w, h = (int(v) for v in np.asarray(data["image_size"]).ravel()[:2])
                size = (w, h)
            rms = None
            for key in ("rms", "rms_error", "rms_reprojection_error_px"):
                if key in keys:
                    rms = float(np.asarray(data[key]).ravel()[0])
                    break
            return camera, dist, size, rms
    if suffix == ".json":
        info = json.loads(path.read_text())
        mat = info.get("camera_matrix", {})
        if isinstance(mat, dict) and "matrix_3x3" in mat:
            camera = np.asarray(mat["matrix_3x3"], dtype=np.float64).reshape(3, 3)
        else:
            fx = float(mat.get("fx", info.get("fx", 0.0)))
            fy = float(mat.get("fy", info.get("fy", 0.0)))
            cx = float(mat.get("cx", info.get("cx", 0.0)))
            cy = float(mat.get("cy", info.get("cy", 0.0)))
            camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray(info.get("distortion_coefficients",
                                   info.get("dist_coeffs", [])), dtype=np.float64).reshape(-1, 1)
        cal = info.get("calibration_info", {}) if isinstance(info.get("calibration_info"), dict) else {}
        size = None
        if "image_width" in cal and "image_height" in cal:
            size = (int(cal["image_width"]), int(cal["image_height"]))
        rms = cal.get("rms_reprojection_error_px", info.get("rms_error"))
        return camera, dist, size, (float(rms) if rms is not None else None)
    if suffix in (".yaml", ".yml"):
        import yaml
        info = yaml.safe_load(path.read_text())
        camera = np.asarray(info["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(info["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
        size = None
        if "image_width" in info and "image_height" in info:
            size = (int(info["image_width"]), int(info["image_height"]))
        return camera, dist, size, None
    raise SystemExit(f"Unsupported calibration file {path}; use one of the calibrator's "
                     f"camera_calibration.npz / camera_calibration.json / camera_info.yaml")


def camera_model(width: int, height: int, calibration: Path | None, fov: float) -> tuple[np.ndarray, np.ndarray]:
    if calibration:
        camera, dist, calib_size, rms = load_calibration_file(calibration)
        if calib_size is not None and tuple(calib_size) != (width, height):
            cal_w, cal_h = calib_size
            camera = camera.copy()
            camera[0, 0] *= width / cal_w
            camera[1, 1] *= height / cal_h
            camera[0, 2] *= width / cal_w
            camera[1, 2] *= height / cal_h
            print(f"Calibration is for {cal_w}x{cal_h} but frames are {width}x{height}; "
                  f"scaled intrinsics (same-aspect approximation).", file=sys.stderr)
        rms_note = f" (rms {rms:.3f}px)" if rms is not None else ""
        print(f"Loaded calibration {calibration}{rms_note}.", file=sys.stderr)
        return camera, np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64), np.zeros((5, 1))


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=float) @ np.asarray(second, dtype=float).T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def solvepnp_pose(corners: np.ndarray, size_m: float, camera: np.ndarray,
                  distortion: np.ndarray, prior_rotation: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Refit pose with true tag size; disambiguate correspondence via prior rotation.

    Mirrors the benchmark-winning backend: tries both object-point conventions
    x 4 cyclic shifts x both corner windings and keeps the candidate closest
    to the detector's own rotation (size-independent), dodging the planar
    180° ambiguity that reprojection error alone cannot resolve.
    """
    half = float(size_m) / 2.0
    variants = [
        np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=np.float64),
        np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64),
    ]
    image = np.ascontiguousarray(np.asarray(corners, dtype=np.float64).reshape(4, 2))
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for obj_points in variants:
        for reverse in (False, True):
            points = image[::-1] if reverse else image
            for shift in range(4):
                ordered = np.ascontiguousarray(np.roll(points, shift, axis=0))
                ok, rvec, tvec = cv2.solvePnP(obj_points, ordered, camera, distortion,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                rotation = cv2.Rodrigues(rvec)[0]
                score = rotation_error_degrees(rotation, prior_rotation) if prior_rotation is not None else 0.0
                projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera, distortion)
                reproj = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - ordered, axis=1)))
                if best is None or score < best[0]:
                    best = (score, rotation, tvec.reshape(3), reproj)
    if best is None:
        return None
    return best[1], best[2], best[3]


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([0.25 / s, (r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s, (r[1, 0] - r[0, 1]) * s])
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        return np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    if r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        return np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
    return np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])


def quaternion_to_rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4))
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class PoseFilter:
    """Per-tag exponential smoother: EMA on translation, NLERP on rotation.

    Kills high-frequency pose jitter. Stale tracks (tag unseen) age out
    after `max_missed` frames so ghost poses don't linger.
    """

    def __init__(self, alpha: float = 0.45, max_missed: int = 5) -> None:
        self.alpha = alpha
        self.max_missed = max_missed
        self._state: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def update(self, marker_id: int, translation: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        if marker_id not in self._state:
            self._state[marker_id] = (translation, np.asarray(rotation, dtype=np.float64), 0)
            return translation, np.asarray(rotation, dtype=np.float64)
        prev_t, prev_r, _ = self._state[marker_id]
        smooth_t = self.alpha * translation + (1.0 - self.alpha) * prev_t
        q_new = rotation_to_quaternion(rotation)
        q_prev = rotation_to_quaternion(prev_r)
        if float(q_new @ q_prev) < 0.0:
            q_new = -q_new
        q_smooth = self.alpha * q_new + (1.0 - self.alpha) * q_prev
        smooth_r = quaternion_to_rotation(q_smooth)
        self._state[marker_id] = (smooth_t, smooth_r, 0)
        return smooth_t, smooth_r

    def mark_seen(self, seen_ids: set[int]) -> None:
        for marker_id in list(self._state):
            if marker_id not in seen_ids:
                prev_t, prev_r, missed = self._state[marker_id]
                if missed + 1 >= self.max_missed:
                    del self._state[marker_id]
                else:
                    self._state[marker_id] = (prev_t, prev_r, missed + 1)


def euler_xyz_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
    # Decompose R = Rz(rz) @ Ry(ry) @ Rx(rx).
    ry = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(ry)) > 1e-7:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = math.atan2(-rotation[1, 2], rotation[1, 1])
        rz = 0.0
    return tuple(math.degrees(v) for v in (rx, ry, rz))


def draw_help(frame: np.ndarray, overlays: Overlays) -> None:
    states = [
        ("B", "box boundary", overlays.box_boundary),
        ("I", "top-right indicator", overlays.top_right_indicator),
        ("P", "x/y/z + rx/ry/rz", overlays.pose),
        ("A", "3D axes", overlays.axes),
        ("H", "shortcuts", overlays.help),
    ]
    y = 24
    for key, label, enabled in states:
        color = (90, 235, 130) if enabled else (135, 135, 135)
        cv2.putText(frame, f"{key}: {label} [{'ON' if enabled else 'OFF'}]", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 1, cv2.LINE_AA)
        y += 23
    cv2.putText(frame, "Q / ESC: quit", (12, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 230, 230), 1, cv2.LINE_AA)


def draw_pose_table(frame: np.ndarray, poses: dict[int, tuple[np.ndarray, np.ndarray]], overlays: Overlays,
                    scroll: dict | None = None) -> np.ndarray:
    """Draw relative poses (mm) in a fixed left panel; Up/Down scrolls long lists."""
    if not overlays.pose:
        return frame
    panel_width = min(560, max(520, frame.shape[1] // 2))
    height = frame.shape[0]
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
    panel[:] = (28, 31, 38)
    cv2.putText(panel, "POSES RELATIVE TO LOWEST ID (mm)", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 250), 2, cv2.LINE_AA)
    columns = [("ID", 18), ("d", 62), ("x", 122), ("y", 192), ("z", 262),
               ("rx°", 332), ("ry°", 392), ("rz°", 452)]
    for label, x_pos in columns:
        cv2.putText(panel, label, (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 185, 210), 1, cv2.LINE_AA)
    cv2.line(panel, (16, 58), (panel_width - 16, 58), (75, 82, 95), 1, cv2.LINE_AA)

    if not poses:
        cv2.putText(panel, "No tags detected", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (165, 175, 190), 1, cv2.LINE_AA)
    else:
        origin_id = min(poses)
        row_height = 26
        top0 = 66
        visible = max(1, (height - 8 - top0) // row_height)
        ids_sorted = sorted(poses)
        top_index = 0
        if scroll is not None:
            top_index = int(np.clip(scroll.get("top", 0), 0, max(0, len(ids_sorted) - visible)))
            scroll["top"] = top_index
        hidden_below = 0
        for row, marker_id in enumerate(ids_sorted[top_index:]):
            origin_rotation, origin_translation = poses[origin_id]
            rotation, translation = poses[marker_id]
            relative_rotation = origin_rotation.T @ rotation
            relative_translation = origin_rotation.T @ (translation - origin_translation)
            rx, ry, rz = euler_xyz_degrees(relative_rotation)
            x, y, z = (float(v) * 1000.0 for v in relative_translation)
            distance = float(np.linalg.norm(relative_translation)) * 1000.0
            top = top0 + row * row_height
            if top + row_height > height - 8:
                hidden_below = len(ids_sorted) - top_index - row
                break
            color = (105, 225, 150) if marker_id == origin_id else (235, 240, 250)
            values = [str(marker_id), f"{distance:.1f}", f"{x:+.1f}", f"{y:+.1f}", f"{z:+.1f}",
                      f"{rx:+.1f}", f"{ry:+.1f}", f"{rz:+.1f}"]
            for value, (_label, x_pos) in zip(values, columns):
                cv2.putText(panel, value, (x_pos, top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            cv2.line(panel, (16, top + row_height - 2), (panel_width - 16, top + row_height - 2), (60, 66, 78), 1, cv2.LINE_AA)
        hints = []
        if top_index > 0:
            hints.append(f"^ {top_index} above")
        if hidden_below > 0:
            hints.append(f"+ {hidden_below} below")
        if hints:
            cv2.putText(panel, " ".join(hints) + " (Up/Down scrolls)", (18, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 175, 190), 1, cv2.LINE_AA)
    return np.hstack((panel, frame))


def annotate(frame: np.ndarray, corners: list[np.ndarray], ids: np.ndarray | None,
             poses: dict[int, tuple[np.ndarray, np.ndarray]], overlays: Overlays,
             axis_length: float, camera: np.ndarray, distortion: np.ndarray) -> None:
    ids_list = [] if ids is None else [int(v) for v in ids.flatten()]
    origin_id = min(poses) if poses else None
    height, width = frame.shape[:2]

    if overlays.box_boundary:
        for marker_corners in corners:
            polygon = np.round(marker_corners.reshape(4, 2)).astype(np.int32)
            cv2.polylines(frame, [polygon], True, (65, 235, 110), 2, cv2.LINE_AA)

    for marker_corners, marker_id in zip(corners, ids_list):
        if overlays.top_right_indicator:
            points = marker_corners.reshape(4, 2).astype(int)
            # pupil-apriltags starts its counter-clockwise corner order at the tag's top-right.
            top_right = tuple(points[0])
            if marker_id not in poses:
                # Detected but dropped by pose estimation/gating: red, explicit.
                color = (60, 60, 235)
                suffix = " ?pose"
            else:
                color = (50, 220, 100) if marker_id == origin_id else (0, 190, 255)
                suffix = " O" if marker_id == origin_id else ""
            cv2.circle(frame, top_right, 7, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{marker_id}{suffix}", (top_right[0] + 9, top_right[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        pose = poses.get(marker_id)
        if pose is None:
            continue
        rotation_cam_tag, translation_cam_tag = pose
        if overlays.axes:
            rvec, _ = cv2.Rodrigues(rotation_cam_tag)
            cv2.drawFrameAxes(frame, camera, distortion, rvec, translation_cam_tag.reshape(3, 1), axis_length, 2)

def main() -> None:
    args = parse_args()
    if args.tag_size <= 0:
        raise SystemExit("--tag-size must be positive")
    marker_sizes = dict(MARKER_SIZES_M)
    for entry in args.marker_size:
        try:
            marker_id_text, size_text = entry.split("=", maxsplit=1)
            marker_id, size = int(marker_id_text), float(size_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid --marker-size {entry!r}; use ID=METRES, such as 3=0.08") from exc
        if marker_id < 0 or size <= 0:
            raise SystemExit("Marker IDs must be non-negative and marker sizes must be positive")
        marker_sizes[marker_id] = size
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source
    capture = cv2.VideoCapture(source)
    ok = False
    frame = None
    if capture.isOpened():
        ok, frame = capture.read()
    if not ok and source == 1:
        capture.release()
        print("Camera 1 is unavailable or returned no frames; trying camera 0.", file=sys.stderr)
        source = 0
        capture = cv2.VideoCapture(source)
        if capture.isOpened():
            ok, frame = capture.read()
    if not ok:
        capture.release()
        raise SystemExit(f"Could not read frames from camera or video source {args.source!r}")

    detector = Detector(
        families=f"tag{args.dictionary}",
        nthreads=2,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    overlays = Overlays()
    pose_filter = PoseFilter(alpha=args.smooth_alpha, max_missed=args.max_missed)
    table_scroll: dict = {"top": 0}
    camera = distortion = None
    cv2.namedWindow("AprilTag relative pose", cv2.WINDOW_NORMAL)

    try:
        while ok:
            height, width = frame.shape[:2]
            if camera is None:
                camera, distortion = camera_model(width, height, args.calibration, args.fov)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intrinsics = (float(camera[0, 0]), float(camera[1, 1]), float(camera[0, 2]), float(camera[1, 2]))
            detections = []
            with suppress_native_noise():
                detections = detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=intrinsics,
                    tag_size=args.tag_size,
                )
            corners = [np.asarray(detection.corners, dtype=np.float32) for detection in detections]
            ids = np.asarray([detection.tag_id for detection in detections], dtype=np.int32)
            poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for detection in detections:
                marker_id = int(detection.tag_id)
                size = marker_sizes.get(marker_id, args.tag_size)
                rotation = np.asarray(detection.pose_R, dtype=np.float64)
                # AprilTag pose translation scales linearly with physical tag edge size.
                translation = np.asarray(detection.pose_t, dtype=np.float64).reshape(3) * (size / args.tag_size)
                if args.pose_backend == "solvepnp":
                    # Refit with the TRUE tag size (benchmarked: ~25% better rotation).
                    refined = solvepnp_pose(np.asarray(detection.corners), size,
                                            camera, distortion, rotation)
                    if refined is None:
                        continue
                    rotation, translation, reproj_err = refined
                    if reproj_err > args.max_reproj_px:
                        continue  # wild fit (e.g. extreme tilt): don't feed the filter
                translation, rotation = pose_filter.update(marker_id, translation, rotation)
                poses[marker_id] = (rotation, translation)
            pose_filter.mark_seen(set(poses))
            annotate(frame, corners, ids, poses, overlays, args.axis_length, camera, distortion)
            origin = min(poses) if poses else None
            n_detected = 0 if ids is None else len(ids)
            n_dropped = n_detected - len(poses)
            state = (f"Tags: {len(poses)} posed/{n_detected} detected | "
                     f"origin: {origin if origin is not None else 'none'}"
                     + (f" | dropped: {n_dropped}" if n_dropped else ""))
            cv2.putText(frame, state, (max(12, width - 285), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            if overlays.help:
                draw_help(frame, overlays)
            display = draw_pose_table(frame, poses, overlays, table_scroll)
            cv2.imshow("AprilTag relative pose", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Viewer closed by keyboard input.", file=sys.stderr)
                break
            elif key == ord("b"):
                overlays.box_boundary = not overlays.box_boundary
            elif key == ord("i"):
                overlays.top_right_indicator = not overlays.top_right_indicator
            elif key == ord("p"):
                overlays.pose = not overlays.pose
            elif key == ord("a"):
                overlays.axes = not overlays.axes
            elif key == ord("h"):
                overlays.help = not overlays.help
            elif key == 82:  # Up arrow: scroll table up
                table_scroll["top"] = max(0, table_scroll["top"] - 1)
            elif key == 84:  # Down arrow: scroll table down
                table_scroll["top"] = table_scroll["top"] + 1
            ok, frame = capture.read()
            if not ok and isinstance(source, int):
                next_source = 0 if source == 1 else source
                print(f"Camera {source} stopped returning frames; reconnecting to camera {next_source}.", file=sys.stderr)
                capture.release()
                source = next_source
                capture = cv2.VideoCapture(source)
                if capture.isOpened():
                    ok, frame = capture.read()
                if not ok:
                    print(f"Could not read from camera {source} after reconnect.", file=sys.stderr)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
