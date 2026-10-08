#!/usr/bin/env python3
"""Render known AprilTag poses in Blender and compare pupil-apriltags estimates."""

from __future__ import annotations

import argparse
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
# OpenCV camera coordinates: x right, y down, z forward.
TAG_POSITIONS_CV = {0: (-0.12, 0.0, 0.80), 4: (0.16, -0.07, 1.05)}
# Shared Blender XYZ rotation (radians), so the relative tag rotation is identity.
TAG_ROTATION_XYZ = (0.06, -0.10, 0.04)


def blender_scene_script(output_dir: Path) -> str:
    """Return the small Blender-side script used to build and render the scene."""
    escaped = str(output_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    return f'''import bpy
import math
import os
from mathutils import Vector

out = r'{escaped}'
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
scene.render.filepath = os.path.join(out, "blender_apriltags.png")
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

specs = [
    (0, 0.095, (-0.12, 0.0, 0.80)),
    (4, 0.06725, (0.16, -0.07, 1.05)),
]
for tag_id, size, cv_position in specs:
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
    obj.rotation_euler = {TAG_ROTATION_XYZ!r}
    obj.data.materials.append(tag_material(os.path.join(out, "tag_%d.png" % tag_id), "Tag %d emission" % tag_id))

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "blender_apriltags.blend"))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender", help="Blender executable (default: blender)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blender_pose_sanity"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id in TAG_SIZES_M:
        marker = cv2.aruco.generateImageMarker(family, tag_id, 512, borderBits=1)
        if not cv2.imwrite(str(output_dir / f"tag_{tag_id}.png"), marker):
            raise RuntimeError(f"Could not save generated tag ID {tag_id}")

    scene_script = output_dir / "build_scene.py"
    scene_script.write_text(blender_scene_script(output_dir), encoding="utf-8")
    render_path = output_dir / "blender_apriltags.png"
    render_path.unlink(missing_ok=True)
    blender_version = subprocess.run(
        [args.blender, "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    subprocess.run(
        [args.blender, "--background", "--python", str(scene_script)],
        check=True,
        cwd=output_dir,
    )

    rendered = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)
    if rendered is None:
        raise RuntimeError("Blender did not create blender_apriltags.png")
    intrinsics = (FOCAL_PIXELS, FOCAL_PIXELS, WIDTH / 2.0, HEIGHT / 2.0)
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    detections = detector.detect(
        rendered,
        estimate_tag_pose=True,
        camera_params=intrinsics,
        tag_size=REFERENCE_SIZE_M,
    )

    detected: dict[int, dict[str, object]] = {}
    estimated_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # OpenCV camera axes are x-right/y-down/z-forward. pupil-apriltags' tag frame
    # uses the tag-local x axis opposite this Blender plane's x axis.
    world_to_camera = np.diag([1.0, -1.0, -1.0])
    pupil_tag_to_blender = np.diag([-1.0, 1.0, -1.0])
    expected_rotation = world_to_camera @ rotation_xyz_matrix(TAG_ROTATION_XYZ) @ pupil_tag_to_blender
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id not in TAG_SIZES_M:
            continue
        size_scale = TAG_SIZES_M[tag_id] / REFERENCE_SIZE_M
        estimated_position = np.asarray(detection.pose_t, dtype=float).reshape(3) * size_scale
        estimated_rotation = np.asarray(detection.pose_R, dtype=float)
        expected_position = np.asarray(TAG_POSITIONS_CV[tag_id], dtype=float)
        delta_rotation = estimated_rotation @ expected_rotation.T
        rotation_error = math.degrees(math.acos(float(np.clip((np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0))))
        detected[tag_id] = {
            "expected_camera_xyz_m": expected_position.tolist(),
            "estimated_camera_xyz_m": estimated_position.tolist(),
            "position_error_m": float(np.linalg.norm(estimated_position - expected_position)),
            "rotation_error_deg": rotation_error,
            "corners_px": np.asarray(detection.corners).round(2).tolist(),
        }
        estimated_poses[tag_id] = (estimated_position, estimated_rotation)

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
    passed = set(detected) == set(TAG_SIZES_M)
    if set(detected) == set(TAG_SIZES_M):
        tag0 = detected[0]
        tag4 = detected[4]
        t0_est, r0_est = estimated_poses[0]
        t4_est, r4_est = estimated_poses[4]
        t0_expected = np.asarray(TAG_POSITIONS_CV[0], dtype=float)
        t4_expected = np.asarray(TAG_POSITIONS_CV[4], dtype=float)
        estimated_rel_t = r0_est.T @ (t4_est - t0_est)
        expected_rel_t = expected_rotation.T @ (t4_expected - t0_expected)
        relative_rotation = r0_est.T @ r4_est
        relative_rotation_error = math.degrees(math.acos(float(np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0
        ))))
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
        report["relative_pose_from_id_0_to_id_4"] = relative_pose
        passed = passed and all(
            float(detected[tag_id]["position_error_m"]) <= MAX_POSITION_ERROR_M
            and float(detected[tag_id]["rotation_error_deg"]) <= MAX_ROTATION_ERROR_DEG
            for tag_id in TAG_SIZES_M
        )
        passed = passed and float(relative_pose["translation_error_m"]) <= MAX_RELATIVE_TRANSLATION_ERROR_M
        passed = passed and float(relative_pose["distance_error_m"]) <= MAX_RELATIVE_DISTANCE_ERROR_M
        passed = passed and float(relative_pose["rotation_error_deg"]) <= MAX_ROTATION_ERROR_DEG
    report["sanity_check_passed"] = bool(passed)
    report_path = output_dir / "pose_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved Blender scene, render, script, tags, and report under {output_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
