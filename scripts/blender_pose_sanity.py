#!/usr/bin/env python3
"""Render known AprilTag poses in Blender and compare pupil-apriltags estimates.

Inputs (render once) and outputs (re-process freely) are separated so future
pipelines can be tested WITHOUT re-rendering:

    experiments/poses_30.csv             # experiment configurations
    experiments/inputs/<exp_id>/
        input.png                        # raw Blender render (frozen input)
        ground_truth.json                # GT poses / intrinsics / tag sizes
        build_scene.py / scene.blend     # reproducibility
    experiments/outputs/<pipeline>/<exp_id>/
        annotated.png                    # detections + projected axes overlay
        report.json                      # per-experiment pose report
    experiments/outputs/<pipeline>/summary.{csv,json}

Single-run mode (backwards compatible):
    blender_pose_sanity.py [--output-dir artifacts/blender_pose_sanity]

Batch render + process (default):
    blender_pose_sanity.py --poses-csv experiments/poses_30.csv \
        --experiments-root experiments [--blender blender]

Render only (no detection, keeps inputs frozen):
    blender_pose_sanity.py --poses-csv experiments/poses_30.csv --stage render

Process only (no Blender; try a different detector config on saved inputs):
    blender_pose_sanity.py --poses-csv experiments/poses_30.csv --stage process \\
        --pipeline baseline_qd05 --quad-decimate 0.5

Helper to create the CSV (poses are FOV-validated, see --fov-margin-px):
    blender_pose_sanity.py --generate-poses-csv experiments/poses_30.csv \\
        --num-experiments 30 [--seed 11]

CSV columns:
    exp_id,tag0_x,tag0_y,tag0_z,tag0_rx,tag0_ry,tag0_rz,
           tag4_x,tag4_y,tag4_z,tag4_rx,tag4_ry,tag4_rz
Positions are OpenCV camera coordinates in metres (x right, y down, z forward).
Rotations are Blender XYZ Euler angles in radians (per-tag orientation).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector


WIDTH, HEIGHT = 640, 480
FOCAL_PIXELS = 500.0
REFERENCE_SIZE_M = 0.05
MAX_POSITION_ERROR_M = 0.03
MAX_RELATIVE_TRANSLATION_ERROR_M = 0.02
MAX_RELATIVE_DISTANCE_ERROR_M = 0.02
MAX_ROTATION_ERROR_DEG = 5.0
TAG_SIZES_M = {0: 0.095, 4: 0.06725}
TAG_IDS = (0, 4)
# OpenCV camera coordinates: x right, y down, z forward (single-run defaults).
TAG_POSITIONS_CV = {0: (-0.12, 0.0, 0.80), 4: (0.16, -0.07, 1.05)}
# Shared Blender XYZ rotation (radians), so the relative tag rotation is identity.
TAG_ROTATION_XYZ = (0.06, -0.10, 0.04)

POSES_CSV_FIELDS = (
    "exp_id",
    "tag0_x", "tag0_y", "tag0_z", "tag0_rx", "tag0_ry", "tag0_rz",
    "tag4_x", "tag4_y", "tag4_z", "tag4_rx", "tag4_ry", "tag4_rz",
)


def blender_scene_script(
    output_dir: Path,
    tags: list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]] | None = None,
    assets_dir: Path | None = None,
    render_name: str = "blender_apriltags.png",
    blend_name: str = "blender_apriltags.blend",
) -> str:
    """Return the Blender-side script used to build and render the scene.

    tags: list of (tag_id, size_m, cv_position_xyz, blender_xyz_euler).
    assets_dir: directory holding the generated tag_N.png textures.
    """
    if tags is None:
        tags = [
            (0, TAG_SIZES_M[0], TAG_POSITIONS_CV[0], TAG_ROTATION_XYZ),
            (4, TAG_SIZES_M[4], TAG_POSITIONS_CV[4], TAG_ROTATION_XYZ),
        ]
    assets = assets_dir.resolve() if assets_dir is not None else output_dir.resolve()
    escaped_out = str(output_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_assets = str(assets).replace("\\", "\\\\").replace("'", "\\'")
    specs_repr = repr([
        (tag_id, size, tuple(pos), tuple(rot)) for tag_id, size, pos, rot in tags
    ])
    return f'''import bpy
import math
import os
from mathutils import Vector

out = r'{escaped_out}'
assets = r'{escaped_assets}'
os.makedirs(out, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {{item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = {WIDTH}
scene.render.resolution_y = {HEIGHT}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.filepath = os.path.join(out, "{render_name}")
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
bpy.context.preferences.filepaths.save_version = 0

world = bpy.data.worlds.new("Neutral background")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.52, 0.52, 0.52, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

camera_data = bpy.data.cameras.new("Sanity camera")
camera_data.lens = {FOCAL_PIXELS} * 36.0 / {WIDTH}
camera_data.sensor_width = 36.0
camera_data.sensor_fit = "HORIZONTAL"
camera_data.clip_start = 0.01
camera_data.clip_end = 10.0
camera = bpy.data.objects.new("Sanity camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 0.0)
camera.rotation_euler = (0.0, 0.0, 0.0)  # Blender cameras look along local -Z.
scene.camera = camera

def tag_material(image_path, name):
    image = bpy.data.images.load(image_path, check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Closest"
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material

specs = {specs_repr}
for tag_id, size, cv_position in [(s[0], s[1], s[2]) for s in specs]:
    pass
for tag_id, size, cv_position, rot_xyz in specs:
    cv_x, cv_y, cv_z = cv_position
    # Blender world coordinates converted from OpenCV camera coordinates.
    center = (cv_x, -cv_y, -cv_z)
    h = size / 2.0
    mesh = bpy.data.meshes.new("AprilTag %d mesh" % tag_id)
    mesh.from_pydata([(-h,-h,0), (h,-h,0), (h,h,0), (-h,h,0)], [], [(0,1,2,3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    for loop_index, uv_coord in zip(mesh.polygons[0].loop_indices, [(0,0),(1,0),(1,1),(0,1)]):
        uv.data[loop_index].uv = uv_coord
    obj = bpy.data.objects.new("AprilTag ID %d" % tag_id, mesh)
    scene.collection.objects.link(obj)
    obj.location = center
    obj.rotation_euler = rot_xyz
    obj.data.materials.append(tag_material(os.path.join(assets, "tag_%d.png" % tag_id), "Tag %d emission" % tag_id))

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "{blend_name}"))
bpy.ops.render.render(write_still=True)
'''


def rotation_xyz_matrix(angles: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = angles
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def rotation_error_deg(r_est: np.ndarray, r_exp: np.ndarray) -> float:
    delta = np.asarray(r_est, dtype=float) @ np.asarray(r_exp, dtype=float).T
    return math.degrees(math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))))


def expected_rotation_cv(rot_xyz: tuple[float, float, float]) -> np.ndarray:
    """Expected pupil-apriltags rotation for a tag with Blender XYZ euler rot_xyz."""
    world_to_camera = np.diag([1.0, -1.0, -1.0])
    pupil_tag_to_blender = np.diag([-1.0, 1.0, -1.0])
    return world_to_camera @ rotation_xyz_matrix(rot_xyz) @ pupil_tag_to_blender


def ensure_tag_images(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id in TAG_SIZES_M:
        marker = cv2.aruco.generateImageMarker(family, tag_id, 512, borderBits=1)
        if not cv2.imwrite(str(cache_dir / f"tag_{tag_id}.png"), marker):
            raise RuntimeError(f"Could not save generated tag ID {tag_id}")


def generate_poses_csv(path: Path, num_experiments: int = 30, seed: int = 11,
                       fov_margin_px: int = 25, min_center_sep_px: float = 90.0,
                       max_tilt_deg: float = 55.0,
                       pos_xy_range: float = 0.05,
                       z0_range: tuple[float, float] = (0.72, 0.88),
                       z4_range: tuple[float, float] = (0.97, 1.13),
                       rot_xy_max: float = 0.45, rot_z_max: float = 0.15,
                       rel_rot_max: float = 0.20) -> Path:
    """Write a deterministic CSV with FOV-validated poses in different orientations.

    Every accepted row is guaranteed (under the pinhole model, no distortion):
      * all 8 tag corners project strictly inside the WxH image with margin,
      * tag plane tilt vs. the camera axis < max_tilt_deg,
      * tag centres are at least min_center_sep_px apart (limits overlap).
    Rejection sampling with the given seed keeps poses deterministic.

    Variability knobs: pos_xy_range (m, around base centres), z0/z4 ranges (m,
    use higher values for long-distance cases), rot_xy_max (rad, per-tag
    pitch/yaw magnitude), rot_z_max (rad roll), rel_rot_max (rad extra
    tag-4-vs-tag-0 rotation so relative poses vary too).
    """
    rng = np.random.default_rng(seed)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    attempts = 0
    while len(rows) < num_experiments:
        attempts += 1
        if attempts > num_experiments * 2000:
            raise RuntimeError("Could not find enough FOV-valid poses; relax margins")
        i = len(rows) + 1
        rx0 = float(rng.uniform(-rot_xy_max, rot_xy_max))
        ry0 = float(rng.uniform(-rot_xy_max, rot_xy_max))
        rz0 = float(rng.uniform(-rot_z_max, rot_z_max))
        # Tag 4 gets its own orientation so the relative pose varies too.
        rx4 = float(rx0 + rng.uniform(-rel_rot_max, rel_rot_max))
        ry4 = float(ry0 + rng.uniform(-rel_rot_max, rel_rot_max))
        rz4 = float(rz0 + rng.uniform(-rel_rot_max / 2, rel_rot_max / 2))
        candidate = {
            "exp_id": f"{i:03d}",
            "tag0_x": round(float(-0.12 + rng.uniform(-pos_xy_range, pos_xy_range)), 4),
            "tag0_y": round(float(0.00 + rng.uniform(-pos_xy_range, pos_xy_range)), 4),
            "tag0_z": round(float(rng.uniform(*z0_range)), 4),
            "tag0_rx": round(rx0, 4), "tag0_ry": round(ry0, 4), "tag0_rz": round(rz0, 4),
            "tag4_x": round(float(0.16 + rng.uniform(-pos_xy_range, pos_xy_range)), 4),
            "tag4_y": round(float(-0.07 + rng.uniform(-pos_xy_range, pos_xy_range)), 4),
            "tag4_z": round(float(rng.uniform(*z4_range)), 4),
            "tag4_rx": round(rx4, 4), "tag4_ry": round(ry4, 4), "tag4_rz": round(rz4, 4),
        }
        ok, _reason = check_specs_in_fov(
            row_to_specs(candidate), margin_px=fov_margin_px,
            min_center_sep_px=min_center_sep_px, max_tilt_deg=max_tilt_deg)
        if ok:
            rows.append(candidate)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=POSES_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def tag_corners_cam(pos: tuple[float, float, float],
                    rot_xyz: tuple[float, float, float],
                    size_m: float) -> np.ndarray:
    """Return the 4 tag corners in OpenCV camera coordinates."""
    h = size_m / 2.0
    corners_tag = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    r_exp = expected_rotation_cv(rot_xyz)
    return np.asarray(pos, dtype=float) + corners_tag @ r_exp.T


def project_points(p_cams: np.ndarray) -> np.ndarray:
    p = np.asarray(p_cams, dtype=float)
    z = np.clip(p[..., 2], 1e-6, None)
    u = FOCAL_PIXELS * p[..., 0] / z + WIDTH / 2.0
    v = FOCAL_PIXELS * p[..., 1] / z + HEIGHT / 2.0
    return np.stack([u, v], axis=-1)


def check_specs_in_fov(specs: list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]],
                       margin_px: int = 25, min_center_sep_px: float = 90.0,
                       max_tilt_deg: float = 55.0) -> tuple[bool, str]:
    """Validate that all tags are fully inside the camera frustum image map.

    Returns (ok, reason). Checks per tag: depth > 0.2 m, tilt limit, and all
    four corners projecting inside [margin, W-margin] x [margin, H-margin].
    Also checks the two tag centres are separated enough to avoid overlap.
    """
    centres_px: list[np.ndarray] = []
    for tag_id, size, pos, rot in specs:
        if pos[2] < 0.20:
            return False, f"tag {tag_id} too close (z={pos[2]:.3f})"
        r_exp = expected_rotation_cv(rot)
        normal_cam = r_exp @ np.array([0.0, 0.0, 1.0])
        tilt_deg = math.degrees(math.acos(float(np.clip(normal_cam[2], -1.0, 1.0))))
        if tilt_deg > max_tilt_deg:
            return False, f"tag {tag_id} tilt {tilt_deg:.1f}deg > {max_tilt_deg}deg"
        corners_cam = tag_corners_cam(pos, rot, size)
        if bool((corners_cam[:, 2] < 0.20).any()):
            return False, f"tag {tag_id} corner behind near plane"
        px = project_points(corners_cam)
        if bool((px[:, 0] < margin_px).any() or (px[:, 0] > WIDTH - margin_px).any()
                or (px[:, 1] < margin_px).any() or (px[:, 1] > HEIGHT - margin_px).any()):
            return False, f"tag {tag_id} corners outside FOV margin {margin_px}px"
        centres_px.append(project_points(np.asarray(pos, dtype=float).reshape(1, 3))[0])
    if len(centres_px) == 2:
        sep = float(np.linalg.norm(centres_px[0] - centres_px[1]))
        if sep < min_center_sep_px:
            return False, f"tag centres {sep:.1f}px apart < {min_center_sep_px}px"
    return True, "ok"


def load_poses_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in POSES_CSV_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Poses CSV {path} is missing columns: {missing}")
        rows = []
        for line in reader:
            if not line.get("exp_id"):
                continue
            rows.append({k: line[k] for k in POSES_CSV_FIELDS})
        return rows


def row_to_specs(row: dict) -> list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]]:
    return [
        (0, TAG_SIZES_M[0],
         (float(row["tag0_x"]), float(row["tag0_y"]), float(row["tag0_z"])),
         (float(row["tag0_rx"]), float(row["tag0_ry"]), float(row["tag0_rz"]))),
        (4, TAG_SIZES_M[4],
         (float(row["tag4_x"]), float(row["tag4_y"]), float(row["tag4_z"])),
         (float(row["tag4_rx"]), float(row["tag4_ry"]), float(row["tag4_rz"]))),
    ]


def project_point(p_cam: np.ndarray) -> tuple[int, int]:
    x, y, z = (float(v) for v in p_cam)
    if z <= 1e-6:
        z = 1e-6
    return int(FOCAL_PIXELS * x / z + WIDTH / 2.0), int(FOCAL_PIXELS * y / z + HEIGHT / 2.0)


def save_annotated_image(
    render_bgr: np.ndarray,
    detections_by_id: dict[int, object],
    estimated_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    detected_report: dict[int, dict[str, object]],
    out_path: Path,
) -> None:
    annotated = render_bgr.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for tag_id, detection in detections_by_id.items():
        corners = np.asarray(detection.corners, dtype=float).reshape(-1, 2).astype(int)
        cv2.polylines(annotated, [corners], True, (0, 255, 0), 2)
        center = tuple(np.asarray(detection.center, dtype=float).astype(int).tolist())
        cv2.circle(annotated, center, 4, (0, 0, 255), -1)
        err = detected_report.get(tag_id, {})
        label = f"id={tag_id} pe={err.get('position_error_m', float('nan')):.3f}m re={err.get('rotation_error_deg', float('nan')):.1f}deg"
        cv2.putText(annotated, label, (int(corners[:, 0].min()), max(15, int(corners[:, 1].min()) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        # Project tag-frame axes (length = half tag size) using the estimated pose.
        if tag_id in estimated_poses:
            t_est, r_est = estimated_poses[tag_id]
            axis_len = TAG_SIZES_M[tag_id] * 0.75
            origin = project_point(t_est)
            for axis, color in zip(np.eye(3) * axis_len, colors):
                tip = project_point(t_est + r_est @ axis)
                cv2.arrowedLine(annotated, origin, tip, color, 2, tipLength=0.15)
    cv2.imwrite(str(out_path), annotated)


def solvepnp_pose(corners_px: np.ndarray, size_m: float,
                  prior_R: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Pose via cv2.solvePnP with correspondence disambiguated by a prior rotation.

    The pupil-apriltags corner order / tag-frame handedness is not assumed:
    y-down and y-up object-point conventions x 4 cyclic shifts are tried, and
    the candidate closest to prior_R (the detector's own rotation, which is
    size-independent) is kept. This dodges the planar-pose 180° ambiguity that
    reprojection error alone cannot resolve. Returns (t, R) or None.
    """
    h = size_m / 2.0
    obj_variants = [
        np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], dtype=np.float64),
        np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64),
    ]
    cam = np.array([[FOCAL_PIXELS, 0, WIDTH / 2.0],
                    [0, FOCAL_PIXELS, HEIGHT / 2.0], [0, 0, 1]], dtype=np.float64)
    img = np.ascontiguousarray(np.asarray(corners_px, dtype=np.float64).reshape(4, 2))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for obj in obj_variants:
        for shift in range(4):
            ordered = np.ascontiguousarray(np.roll(img, shift, axis=0))
            ok, rvec, tvec = cv2.solvePnP(obj, ordered, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            R = cv2.Rodrigues(rvec)[0]
            score = rotation_error_deg(R, prior_R) if prior_R is not None else 0.0
            if best is None or score < best[0]:
                best = (score, tvec.reshape(3), R)
    if best is None:
        return None
    return best[1], best[2]


def detect_pose(detection, pose_backend: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate (t, R) for one detection with the chosen pose backend."""
    tag_id = int(detection.tag_id)
    if pose_backend == "solvepnp":
        return solvepnp_pose(np.asarray(detection.corners), TAG_SIZES_M[tag_id],
                             np.asarray(detection.pose_R, dtype=float))
    size_scale = TAG_SIZES_M[tag_id] / REFERENCE_SIZE_M
    t = np.asarray(detection.pose_t, dtype=float).reshape(3) * size_scale
    return t, np.asarray(detection.pose_R, dtype=float)


def evaluate_specs(
    specs: list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]],
    detections: list,
    pose_backend: str = "tag_pose",
) -> tuple[dict[int, dict[str, object]], dict[int, tuple[np.ndarray, np.ndarray]], dict[int, object]]:
    expected_pos = {tag_id: np.asarray(pos, dtype=float) for tag_id, _, pos, _ in specs}
    expected_rot = {tag_id: expected_rotation_cv(rot) for tag_id, _, _, rot in specs}
    detected: dict[int, dict[str, object]] = {}
    estimated_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    detections_by_id: dict[int, object] = {}
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id not in TAG_SIZES_M:
            continue
        if tag_id in detections_by_id:
            continue
        pose = detect_pose(detection, pose_backend)
        if pose is None:
            continue
        estimated_position, estimated_rotation = pose
        rotation_error = rotation_error_deg(estimated_rotation, expected_rot[tag_id])
        detected[tag_id] = {
            "expected_camera_xyz_m": expected_pos[tag_id].tolist(),
            "pose_backend": pose_backend,
            "estimated_camera_xyz_m": estimated_position.tolist(),
            "position_error_m": float(np.linalg.norm(estimated_position - expected_pos[tag_id])),
            "rotation_error_deg": rotation_error,
            "corners_px": np.asarray(detection.corners).round(2).tolist(),
        }
        estimated_poses[tag_id] = (estimated_position, estimated_rotation)
        detections_by_id[tag_id] = detection
    # Fill in expected euler angles (kept separate for clarity).
    euler_by_id = {tag_id: tuple(rot) for tag_id, _, _, rot in specs}
    for tag_id in detected:
        detected[tag_id]["expected_blender_euler_xyz_rad"] = [float(v) for v in euler_by_id[tag_id]]
    return detected, estimated_poses, detections_by_id


def check_thresholds(
    detected: dict[int, dict[str, object]],
    estimated_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    specs: list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]],
) -> tuple[bool, dict[str, object] | None]:
    passed = set(detected) == set(TAG_SIZES_M)
    relative_pose: dict[str, object] | None = None
    if set(detected) == set(TAG_SIZES_M):
        expected_pos = {tag_id: np.asarray(pos, dtype=float) for tag_id, _, pos, _ in specs}
        expected_rot = {tag_id: expected_rotation_cv(rot) for tag_id, _, _, rot in specs}
        t0_est, r0_est = estimated_poses[0]
        t4_est, r4_est = estimated_poses[4]
        estimated_rel_t = r0_est.T @ (t4_est - t0_est)
        expected_rel_t = expected_rot[0].T @ (expected_pos[4] - expected_pos[0])
        relative_rotation = r0_est.T @ r4_est
        expected_rel_rotation = expected_rot[0].T @ expected_rot[4]
        relative_rotation_error = rotation_error_deg(relative_rotation, expected_rel_rotation)
        expected_rel_distance = float(np.linalg.norm(expected_rel_t))
        estimated_rel_distance = float(np.linalg.norm(estimated_rel_t))
        relative_pose = {
            "expected_xyz_m": expected_rel_t.tolist(),
            "estimated_xyz_m": estimated_rel_t.tolist(),
            "translation_error_m": float(np.linalg.norm(estimated_rel_t - expected_rel_t)),
            "expected_distance_m": expected_rel_distance,
            "estimated_distance_m": estimated_rel_distance,
            "distance_error_m": abs(estimated_rel_distance - expected_rel_distance),
            "rotation_error_deg": relative_rotation_error,
        }
        passed = passed and all(
            float(detected[tag_id]["position_error_m"]) <= MAX_POSITION_ERROR_M
            and float(detected[tag_id]["rotation_error_deg"]) <= MAX_ROTATION_ERROR_DEG
            for tag_id in TAG_SIZES_M
        )
        passed = passed and float(relative_pose["translation_error_m"]) <= MAX_RELATIVE_TRANSLATION_ERROR_M
        passed = passed and float(relative_pose["distance_error_m"]) <= MAX_RELATIVE_DISTANCE_ERROR_M
        passed = passed and float(relative_pose["rotation_error_deg"]) <= MAX_ROTATION_ERROR_DEG
    return bool(passed), relative_pose


def ground_truth_payload(exp_id: str,
                         specs: list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]]
                         ) -> dict[str, object]:
    """Ground-truth map payload frozen at render time (inputs side)."""
    px: dict[str, object] = {}
    for tag_id, size, pos, rot in specs:
        corners_cam = tag_corners_cam(pos, rot, size)
        px[str(tag_id)] = project_points(corners_cam).round(2).tolist()
    return {
        "exp_id": exp_id,
        "family": "tag36h11",
        "image": {"width": WIDTH, "height": HEIGHT},
        "intrinsics_px": {"fx": FOCAL_PIXELS, "fy": FOCAL_PIXELS, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0},
        "configured_tag_sizes_m": {str(k): v for k, v in TAG_SIZES_M.items()},
        "tags": [
            {"id": tag_id, "size_m": size,
             "expected_camera_xyz_m": list(pos), "blender_euler_xyz_rad": list(rot)}
            for tag_id, size, pos, rot in specs
        ],
        "projected_corners_px": px,
    }


def render_experiment(exp_id: str,
                      specs: list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]],
                      inputs_exp_dir: Path,
                      assets_dir: Path,
                      blender_exe: str,
                      fov_margin_px: int = 25, min_center_sep_px: float = 90.0,
                      max_tilt_deg: float = 55.0) -> dict[str, object]:
    """STAGE 1 (Blender): render the frozen input image + ground-truth map.

    Writes inputs/<exp_id>/{input.png, ground_truth.json, build_scene.py, scene.blend}.
    Same tag textures (tag_0.png / tag_4.png from assets_dir) are reused.
    """
    ok, reason = check_specs_in_fov(specs, margin_px=fov_margin_px,
                                    min_center_sep_px=min_center_sep_px,
                                    max_tilt_deg=max_tilt_deg)
    if not ok:
        raise ValueError(f"[{exp_id}] pose outside camera FOV map: {reason}")
    inputs_exp_dir.mkdir(parents=True, exist_ok=True)
    (inputs_exp_dir / "build_scene.py").write_text(
        blender_scene_script(inputs_exp_dir, tags=specs, assets_dir=assets_dir,
                             render_name="input.png", blend_name="scene.blend"),
        encoding="utf-8",
    )
    render_path = inputs_exp_dir / "input.png"
    render_path.unlink(missing_ok=True)
    subprocess.run([blender_exe, "--background", "--python",
                    str(inputs_exp_dir / "build_scene.py")],
                   check=True, cwd=inputs_exp_dir, capture_output=True)
    if not render_path.exists():
        raise RuntimeError(f"Blender did not create {render_path}")
    gt = ground_truth_payload(exp_id, specs)
    (inputs_exp_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")
    return gt


def process_experiment(exp_id: str,
                       inputs_exp_dir: Path,
                       outputs_exp_dir: Path,
                       detector: Detector,
                       pipeline: str,
                       pipeline_config: dict[str, object],
                       method: str,
                       pose_backend: str = "tag_pose") -> dict[str, object]:
    """STAGE 2 (no Blender): run a processing pipeline on a frozen input image.

    Reads inputs/<exp_id>/{input.png, ground_truth.json}, writes
    outputs/<pipeline>/<exp_id>/{annotated.png, report.json}.
    """
    render_path = inputs_exp_dir / "input.png"
    gt_path = inputs_exp_dir / "ground_truth.json"
    if not render_path.exists():
        raise FileNotFoundError(f"Missing frozen input {render_path}; run --stage render first")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth map {gt_path}; run --stage render first")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    specs = [
        (int(t["id"]), float(t["size_m"]),
         tuple(t["expected_camera_xyz_m"]), tuple(t["blender_euler_xyz_rad"]))
        for t in gt["tags"]
    ]
    render_bgr = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    rendered = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)
    if rendered is None or render_bgr is None:
        raise RuntimeError(f"Could not read frozen input {render_path}")
    intrinsics = (FOCAL_PIXELS, FOCAL_PIXELS, WIDTH / 2.0, HEIGHT / 2.0)
    detections = detector.detect(
        rendered, estimate_tag_pose=True,
        camera_params=intrinsics, tag_size=REFERENCE_SIZE_M,
    )
    detected, estimated_poses, detections_by_id = evaluate_specs(specs, detections, pose_backend)
    passed, relative_pose = check_thresholds(detected, estimated_poses, specs)
    outputs_exp_dir.mkdir(parents=True, exist_ok=True)
    save_annotated_image(render_bgr, detections_by_id, estimated_poses, detected,
                         outputs_exp_dir / "annotated.png")
    report: dict[str, object] = {
        "exp_id": exp_id,
        "pipeline": pipeline,
        "pipeline_config": pipeline_config,
        "method": method,
        "family": "tag36h11",
        "intrinsics_px": {"fx": FOCAL_PIXELS, "fy": FOCAL_PIXELS, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0},
        "configured_tag_sizes_m": {str(k): v for k, v in TAG_SIZES_M.items()},
        "tags": gt["tags"],
        "frozen_input": str(render_path),
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
    if relative_pose is not None:
        report["relative_pose_from_id_0_to_id_4"] = relative_pose
    (outputs_exp_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_single_experiment(
    exp_id: str,
    specs: list[tuple[int, float, tuple[float, float, float], tuple[float, float, float]]],
    exp_dir: Path,
    assets_dir: Path,
    blender_exe: str,
    blender_version: str,
    detector: Detector,
) -> dict[str, object]:
    """Legacy-shaped helper: render into exp_dir then process in place (kept for compat)."""
    gt = render_experiment(exp_id, specs, exp_dir, assets_dir, blender_exe)
    specs_rt = [
        (int(t["id"]), float(t["size_m"]),
         tuple(t["expected_camera_xyz_m"]), tuple(t["blender_euler_xyz_rad"]))
        for t in gt["tags"]
    ]
    return process_experiment(exp_id, exp_dir, exp_dir, detector, "legacy",
                              {"quad_decimate": 1.0, "nthreads": 2},
                              f"{blender_version} render; pupil-apriltags pose; no lens distortion")


def run_batch(poses_csv: Path, experiments_root: Path, assets_dir: Path, blender_exe: str,
              stage: str = "both", pipeline: str = "baseline",
              quad_decimate: float = 1.0, nthreads: int = 2,
              quad_sigma: float = 0.0, refine_edges: int = 1,
              decode_sharpening: float = 0.25, pose_backend: str = "tag_pose",
              fov_margin_px: int = 25, min_center_sep_px: float = 90.0,
              max_tilt_deg: float = 55.0) -> int:
    """Run render and/or process stages with separated inputs/outputs layout."""
    rows = load_poses_csv(poses_csv)
    if not rows:
        raise ValueError(f"No experiments found in {poses_csv}")
    inputs_root = experiments_root / "inputs"
    pipeline_root = experiments_root / "outputs" / pipeline
    inputs_root.mkdir(parents=True, exist_ok=True)
    # Camera-FOV map check BEFORE any expensive Blender call.
    for row in rows:
        exp_id = str(row["exp_id"]).zfill(3) if str(row["exp_id"]).isdigit() else str(row["exp_id"])
        ok, reason = check_specs_in_fov(row_to_specs(row), margin_px=fov_margin_px,
                                        min_center_sep_px=min_center_sep_px,
                                        max_tilt_deg=max_tilt_deg)
        if not ok:
            raise ValueError(f"[{exp_id}] pose outside camera FOV map: {reason} (fix CSV first)")
    print(f"FOV map check passed for {len(rows)} experiments "
          f"(640x{HEIGHT}, fx={FOCAL_PIXELS}, margin=25px).")
    pipeline_config: dict[str, object] = {
        "pipeline": pipeline, "families": "tag36h11",
        "quad_decimate": quad_decimate, "quad_sigma": quad_sigma,
        "refine_edges": refine_edges, "decode_sharpening": decode_sharpening,
        "pose_backend": pose_backend, "nthreads": nthreads,
        "tag_size_ref_m": REFERENCE_SIZE_M,
    }
    if stage in ("both", "render"):
        ensure_tag_images(assets_dir)
        for row in rows:
            exp_id = str(row["exp_id"]).zfill(3) if str(row["exp_id"]).isdigit() else str(row["exp_id"])
            print(f"[{exp_id}] rendering (same tags) ...")
            render_experiment(exp_id, row_to_specs(row), inputs_root / exp_id,
                              assets_dir, blender_exe, fov_margin_px,
                              min_center_sep_px, max_tilt_deg)
            print(f"[{exp_id}] input frozen -> {inputs_root / exp_id / 'input.png'}")
        write_inputs_map(experiments_root, rows)
    if stage in ("both", "process"):
        detector = Detector(families="tag36h11", nthreads=nthreads, quad_decimate=quad_decimate,
                            quad_sigma=quad_sigma, refine_edges=refine_edges,
                            decode_sharpening=decode_sharpening)
        try:
            blender_version = subprocess.run(
                [blender_exe, "--version"], capture_output=True, text=True, check=True
            ).stdout.splitlines()[0]
        except Exception:
            blender_version = "blender (version unknown at process time)"
        method = (f"{blender_version} render (frozen inputs); pupil-apriltags detect "
                  f"[pipeline={pipeline} qd={quad_decimate} qs={quad_sigma} "
                  f"refine={refine_edges} sharp={decode_sharpening} pose={pose_backend}]; "
                  f"no lens distortion")
        summary_rows: list[dict[str, object]] = []
        all_passed = True
        for row in rows:
            exp_id = str(row["exp_id"]).zfill(3) if str(row["exp_id"]).isdigit() else str(row["exp_id"])
            print(f"[{exp_id}] processing pipeline={pipeline} ...")
            report = process_experiment(exp_id, inputs_root / exp_id,
                                        pipeline_root / exp_id, detector,
                                        pipeline, pipeline_config, method, pose_backend)
            rel = report.get("relative_pose_from_id_0_to_id_4", {})
            summary_rows.append({
                "exp_id": exp_id,
                "passed": bool(report["sanity_check_passed"]),
                "detected_ids": ",".join(sorted(str(k) for k in report["detections"])),
                "tag0_pos_err_m": report["detections"].get("0", {}).get("position_error_m", ""),
                "tag0_rot_err_deg": report["detections"].get("0", {}).get("rotation_error_deg", ""),
                "tag4_pos_err_m": report["detections"].get("4", {}).get("position_error_m", ""),
                "tag4_rot_err_deg": report["detections"].get("4", {}).get("rotation_error_deg", ""),
                "rel_trans_err_m": rel.get("translation_error_m", "") if isinstance(rel, dict) else "",
                "rel_dist_err_m": rel.get("distance_error_m", "") if isinstance(rel, dict) else "",
                "rel_rot_err_deg": rel.get("rotation_error_deg", "") if isinstance(rel, dict) else "",
            })
            all_passed = all_passed and bool(report["sanity_check_passed"])
            print(f"[{exp_id}] passed={report['sanity_check_passed']} "
                  f"-> {pipeline_root / exp_id / 'annotated.png'}, "
                  f"{pipeline_root / exp_id / 'report.json'}")
        pipeline_root.mkdir(parents=True, exist_ok=True)
        with open(pipeline_root / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        (pipeline_root / "summary.json").write_text(
            json.dumps({"pipeline": pipeline, "pipeline_config": pipeline_config,
                        "num_experiments": len(summary_rows), "all_passed": all_passed,
                        "results": summary_rows}, indent=2) + "\n", encoding="utf-8")
        print(f"Batch done: {len(summary_rows)} experiments, pipeline={pipeline}, "
              f"all_passed={all_passed} under {pipeline_root}")
        print(f"Re-run another config later with: --stage process --pipeline <name> "
              f"--quad-decimate <v>  (no re-render needed)")
        return 0 if all_passed else 1
    print(f"Render stage done: {len(rows)} frozen inputs under {inputs_root}")
    return 0


def write_inputs_map(experiments_root: Path, rows: list[dict]) -> Path:
    """Write the ground-truth map aggregating every experiment (inputs side)."""
    out = experiments_root / "inputs" / "map.json"
    out.write_text(json.dumps({
        "image": {"width": WIDTH, "height": HEIGHT},
        "intrinsics_px": {"fx": FOCAL_PIXELS, "fy": FOCAL_PIXELS, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0},
        "configured_tag_sizes_m": {str(k): v for k, v in TAG_SIZES_M.items()},
        "experiments": {
            (str(r["exp_id"]).zfill(3) if str(r["exp_id"]).isdigit() else str(r["exp_id"])): {
                "tags": [
                    {"id": tag_id, "size_m": size,
                     "expected_camera_xyz_m": list(pos), "blender_euler_xyz_rad": list(rot)}
                    for tag_id, size, pos, rot in row_to_specs(r)
                ]
            } for r in rows
        },
    }, indent=2) + "\n", encoding="utf-8")
    return out


def run_legacy_single(output_dir: Path, blender_exe: str) -> int:
    """Original single-scene behaviour, kept for backwards compatibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_tag_images(output_dir)
    specs = [
        (0, TAG_SIZES_M[0], TAG_POSITIONS_CV[0], TAG_ROTATION_XYZ),
        (4, TAG_SIZES_M[4], TAG_POSITIONS_CV[4], TAG_ROTATION_XYZ),
    ]
    blender_version = subprocess.run(
        [blender_exe, "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    # Reuse the batch runner pieces but keep legacy filenames.
    scene_script = output_dir / "build_scene.py"
    scene_script.write_text(blender_scene_script(output_dir), encoding="utf-8")
    render_path = output_dir / "blender_apriltags.png"
    render_path.unlink(missing_ok=True)
    subprocess.run([blender_exe, "--background", "--python", str(scene_script)],
                   check=True, cwd=output_dir)
    render_bgr = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    rendered = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)
    if rendered is None:
        raise RuntimeError("Blender did not create blender_apriltags.png")
    intrinsics = (FOCAL_PIXELS, FOCAL_PIXELS, WIDTH / 2.0, HEIGHT / 2.0)
    detections = detector.detect(rendered, estimate_tag_pose=True,
                                 camera_params=intrinsics, tag_size=REFERENCE_SIZE_M)
    detected, estimated_poses, detections_by_id = evaluate_specs(specs, detections)
    passed, relative_pose = check_thresholds(detected, estimated_poses, specs)
    if render_bgr is not None:
        save_annotated_image(render_bgr, detections_by_id, estimated_poses, detected,
                             output_dir / "annotated.png")
    report: dict[str, object] = {
        "method": f"{blender_version} render; pupil-apriltags pose; no lens distortion",
        "family": "tag36h11",
        "intrinsics_px": {"fx": FOCAL_PIXELS, "fy": FOCAL_PIXELS, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0},
        "configured_tag_sizes_m": TAG_SIZES_M,
        "detections": detected,
        "all_expected_ids_detected": set(detected) == set(TAG_SIZES_M),
        "thresholds": {
            "max_position_error_m": MAX_POSITION_ERROR_M,
            "max_relative_translation_error_m": MAX_RELATIVE_TRANSLATION_ERROR_M,
            "max_relative_distance_error_m": MAX_RELATIVE_DISTANCE_ERROR_M,
            "max_rotation_error_deg": MAX_ROTATION_ERROR_DEG,
        },
    }
    if relative_pose is not None:
        report["relative_pose_from_id_0_to_id_4"] = relative_pose
    report["sanity_check_passed"] = bool(passed)
    (output_dir / "pose_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved Blender scene, render, script, tags, and report under {output_dir}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender", help="Blender executable (default: blender)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blender_pose_sanity"),
                        help="Single-run output dir (legacy mode)")
    parser.add_argument("--poses-csv", type=Path, default=None,
                        help="CSV with per-experiment poses; enables batch mode")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"),
                        help="Batch root: inputs/<exp_id>/input.png + ground_truth.json, "
                             "outputs/<pipeline>/<exp_id>/{annotated.png,report.json}")
    parser.add_argument("--tag-cache-dir", type=Path, default=None,
                        help="Directory holding the SAME tag_N.png textures, reused by every "
                             "experiment (default: <experiments-root>/inputs/_tags, falling "
                             "back to --output-dir for reuse)")
    parser.add_argument("--stage", choices=["both", "render", "process"], default="both",
                        help="both: render inputs then process; render: only Blender inputs; "
                             "process: only run pipeline on frozen inputs (no Blender)")
    parser.add_argument("--pipeline", default="baseline",
                        help="Output pipeline name: experiments/outputs/<pipeline>/ (so different "
                             "processing configs can be compared without re-rendering)")
    parser.add_argument("--quad-decimate", type=float, default=1.0,
                        help="pupil-apriltags quad_decimate for the process stage (try 1.0 vs 0.5)")
    parser.add_argument("--quad-sigma", type=float, default=0.0,
                        help="pupil-apriltags quad_sigma (Gaussian blur on decimated image)")
    parser.add_argument("--refine-edges", type=int, default=1, choices=[0, 1],
                        help="pupil-apriltags edge refinement toggle")
    parser.add_argument("--decode-sharpening", type=float, default=0.25,
                        help="pupil-apriltags decode sharpening")
    parser.add_argument("--pose-backend", default="tag_pose", choices=["tag_pose", "solvepnp"],
                        help="tag_pose: detector pose scaled to true size; "
                             "solvepnp: cv2.solvePnP on corners with true tag size")
    parser.add_argument("--nthreads", type=int, default=2)
    parser.add_argument("--fov-margin-px", type=int, default=25,
                        help="FOV guard margin used when generating/validating the poses CSV")
    parser.add_argument("--min-center-sep-px", type=float, default=90.0)
    parser.add_argument("--max-tilt-deg", type=float, default=55.0)
    parser.add_argument("--pos-xy-range", type=float, default=0.05,
                        help="XY position spread (m) around base centres for generated poses")
    parser.add_argument("--z0-range", type=float, nargs=2, default=(0.72, 0.88),
                        metavar=("Z0_MIN", "Z0_MAX"),
                        help="Depth range (m) for tag 0 (higher = longer distance)")
    parser.add_argument("--z4-range", type=float, nargs=2, default=(0.97, 1.13),
                        metavar=("Z4_MIN", "Z4_MAX"))
    parser.add_argument("--rot-xy-max", type=float, default=0.45,
                        help="Max per-tag pitch/yaw magnitude (rad) for generated poses")
    parser.add_argument("--rot-z-max", type=float, default=0.15)
    parser.add_argument("--rel-rot-max", type=float, default=0.20,
                        help="Max extra tag-4-vs-tag-0 rotation (rad)")
    parser.add_argument("--generate-poses-csv", type=Path, default=None,
                        help="Generate a poses CSV with different orientations and exit")
    parser.add_argument("--num-experiments", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if args.generate_poses_csv is not None:
        out = generate_poses_csv(args.generate_poses_csv, args.num_experiments, args.seed,
                                 fov_margin_px=args.fov_margin_px,
                                 min_center_sep_px=args.min_center_sep_px,
                                 max_tilt_deg=args.max_tilt_deg,
                                 pos_xy_range=args.pos_xy_range,
                                 z0_range=tuple(args.z0_range),
                                 z4_range=tuple(args.z4_range),
                                 rot_xy_max=args.rot_xy_max,
                                 rot_z_max=args.rot_z_max,
                                 rel_rot_max=args.rel_rot_max)
        print(f"Wrote {args.num_experiments} FOV-validated poses to {out}")
        if args.poses_csv is None:
            return 0

    if args.poses_csv is not None:
        assets = args.tag_cache_dir
        if assets is None:
            candidate = args.experiments_root / "inputs" / "_tags"
            assets = candidate
            # Reuse legacy tag images if the new cache is empty (same tags everywhere).
            legacy_tags = list(args.output_dir.glob("tag_*.png"))
            if legacy_tags and not any(candidate.glob("tag_*.png")):
                assets = args.output_dir
        return run_batch(args.poses_csv.resolve(), args.experiments_root.resolve(),
                         assets.resolve(), args.blender, stage=args.stage,
                         pipeline=args.pipeline, quad_decimate=args.quad_decimate,
                         nthreads=args.nthreads, quad_sigma=args.quad_sigma,
                         refine_edges=args.refine_edges,
                         decode_sharpening=args.decode_sharpening,
                         pose_backend=args.pose_backend,
                         fov_margin_px=args.fov_margin_px,
                         min_center_sep_px=args.min_center_sep_px,
                         max_tilt_deg=args.max_tilt_deg)

    return run_legacy_single(args.output_dir.resolve(), args.blender)


if __name__ == "__main__":
    raise SystemExit(main())
