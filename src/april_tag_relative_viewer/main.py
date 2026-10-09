"""Live AprilTag pose viewer with the lowest visible ID as the origin."""

from __future__ import annotations

import argparse
import math
import sys
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
MARKER_SIZES_M: dict[int, float] = {
    0: 0.095,
    1: 0.095,
    2: 0.095,
    3: 0.095,
    4: 0.06725,
    5: 0.06725,
    6: 0.06725,
    7: 0.06725,
    8: 0.06725,
    9: 0.06725,
    10: 0.06725,
    11: 0.06725,
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
    parser.add_argument("source", nargs="?", default="1", help="camera index, video file, or image file (default: 1)")
    parser.add_argument("--tag-size", type=float, default=0.05, help="printed tag edge length in metres (default: 0.05)")
    parser.add_argument("--marker-size", action="append", default=[], metavar="ID=METRES",
                        help="per-ID edge length override; repeat for multiple IDs, e.g. --marker-size 3=0.08")
    parser.add_argument("--dictionary", default="36h11", choices=("16h5", "25h9", "36h11"))
    parser.add_argument("--calibration", type=Path, help="NPZ with camera_matrix and dist_coeffs arrays")
    parser.add_argument("--fov", type=float, default=60.0, help="approximate horizontal camera FOV in degrees without calibration")
    parser.add_argument("--axis-length", type=float, default=0.03, help="drawn axis length in metres")
    parser.add_argument("--quad-decimate", type=float, default=BEST_QUAD_DECIMATE,
                        help="detector decimation (best benchmarked: %(default)s)")
    parser.add_argument("--quad-sigma", type=float, default=BEST_QUAD_SIGMA,
                        help="detector blur sigma (best benchmarked: %(default)s)")
    parser.add_argument("--pose-backend", default="solvepnp", choices=("solvepnp", "tag_pose"),
                        help="solvepnp: cv2.solvePnP with true tag size (best benchmarked); "
                             "tag_pose: detector pose scaled to tag size")
    return parser.parse_args()


def camera_model(width: int, height: int, calibration: Path | None, fov: float) -> tuple[np.ndarray, np.ndarray]:
    if calibration:
        with np.load(calibration) as data:
            return np.asarray(data["camera_matrix"], dtype=np.float64), np.asarray(data["dist_coeffs"], dtype=np.float64)
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
    best: tuple[float, np.ndarray, np.ndarray] | None = None
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
                if best is None or score < best[0]:
                    best = (score, rotation, tvec.reshape(3))
    if best is None:
        return None
    return best[1], best[2]


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


def draw_pose_table(frame: np.ndarray, poses: dict[int, tuple[np.ndarray, np.ndarray]], overlays: Overlays) -> np.ndarray:
    """Draw relative poses in a fixed left panel and return the camera image area."""
    if not overlays.pose:
        return frame
    panel_width = min(560, max(520, frame.shape[1] // 2))
    height = frame.shape[0]
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
    panel[:] = (28, 31, 38)
    cv2.putText(panel, "POSES RELATIVE TO LOWEST ID", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 240, 250), 2, cv2.LINE_AA)
    columns = [("ID", 18), ("d(m)", 70), ("x(m)", 135), ("y(m)", 200), ("z(m)", 265),
               ("rx°", 330), ("ry°", 395), ("rz°", 460)]
    for label, x_pos in columns:
        cv2.putText(panel, label, (x_pos, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 185, 210), 1, cv2.LINE_AA)
    cv2.line(panel, (16, 68), (panel_width - 16, 68), (75, 82, 95), 1, cv2.LINE_AA)

    if not poses:
        cv2.putText(panel, "No tags detected", (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (165, 175, 190), 1, cv2.LINE_AA)
    else:
        origin_id = min(poses)
        row_height = 34
        for row, marker_id in enumerate(sorted(poses)):
            origin_rotation, origin_translation = poses[origin_id]
            rotation, translation = poses[marker_id]
            relative_rotation = origin_rotation.T @ rotation
            relative_translation = origin_rotation.T @ (translation - origin_translation)
            rx, ry, rz = euler_xyz_degrees(relative_rotation)
            x, y, z = relative_translation
            distance = float(np.linalg.norm(relative_translation))
            top = 78 + row * row_height
            if top + row_height > height - 8:
                cv2.putText(panel, f"+ {len(poses) - row} more tags", (18, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (165, 175, 190), 1, cv2.LINE_AA)
                break
            color = (105, 225, 150) if marker_id == origin_id else (235, 240, 250)
            values = [str(marker_id), f"{distance:.3f}", f"{x:+.3f}", f"{y:+.3f}", f"{z:+.3f}",
                      f"{rx:+.1f}", f"{ry:+.1f}", f"{rz:+.1f}"]
            for value, (_label, x_pos) in zip(values, columns):
                cv2.putText(panel, value, (x_pos, top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)
            cv2.line(panel, (16, top + row_height - 3), (panel_width - 16, top + row_height - 3), (60, 66, 78), 1, cv2.LINE_AA)
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
            color = (50, 220, 100) if marker_id == origin_id else (0, 190, 255)
            cv2.circle(frame, top_right, 7, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{marker_id}{' O' if marker_id == origin_id else ''}", (top_right[0] + 9, top_right[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

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
    camera = distortion = None
    cv2.namedWindow("AprilTag relative pose", cv2.WINDOW_NORMAL)

    try:
        while ok:
            height, width = frame.shape[:2]
            if camera is None:
                camera, distortion = camera_model(width, height, args.calibration, args.fov)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intrinsics = (float(camera[0, 0]), float(camera[1, 1]), float(camera[0, 2]), float(camera[1, 2]))
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
                    if refined is not None:
                        rotation, translation = refined
                poses[marker_id] = (rotation, translation)
            annotate(frame, corners, ids, poses, overlays, args.axis_length, camera, distortion)
            origin = min(poses) if poses else None
            state = f"Tags: {len(poses)} | origin: {origin if origin is not None else 'none'}"
            cv2.putText(frame, state, (max(12, width - 285), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            if overlays.help:
                draw_help(frame, overlays)
            display = draw_pose_table(frame, poses, overlays)
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
