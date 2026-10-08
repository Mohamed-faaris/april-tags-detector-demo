"""Diagnose OpenCV's worst frames: compare its corners against pupil's."""
import json
import sys
import glob
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_pose_sanity import (
    FOCAL_PIXELS, HEIGHT, TAG_SIZES_M, WIDTH,
    expected_rotation_cv, rotation_error_deg,
)
from pupil_apriltags import Detector

CAM = np.array([[FOCAL_PIXELS, 0, WIDTH / 2.0],
                [0, FOCAL_PIXELS, HEIGHT / 2.0], [0, 0, 1.]])


def obj_variants(size):
    h = size / 2
    return [np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0.]]),
            np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0.]])]


def best_rot_err(corners, size, rexp):
    best = 1e9
    for obj in obj_variants(size):
        for rev in (False, True):
            pts = corners[::-1] if rev else corners
            for sh in range(4):
                o = np.ascontiguousarray(np.roll(pts, sh, axis=0))
                ok, rv, _ = cv2.solvePnP(obj, o, CAM, None, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    best = min(best, rotation_error_deg(cv2.Rodrigues(rv)[0], rexp))
    return best


fam = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
par = cv2.aruco.DetectorParameters()
par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
ocv = cv2.aruco.ArucoDetector(fam, par)
pd = Detector(families="tag36h11", nthreads=2, quad_decimate=0.5)

for f in sorted(glob.glob("experiments/inputs/[0-9]*/input.png")):
    exp = Path(f).parent.name
    gt = json.loads(Path(f).with_name("ground_truth.json").read_text())
    spec = {int(t["id"]): tuple(t["blender_euler_xyz_rad"]) for t in gt["tags"]}
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    pc = {int(d.tag_id): np.asarray(d.corners).reshape(4, 2) for d in pd.detect(img)}
    corners, ids, _ = ocv.detectMarkers(img)
    oc = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())} if ids is not None else {}
    if 4 in oc and 4 in pc:
        rexp = expected_rotation_cv(spec[4])
        e = best_rot_err(oc[4], TAG_SIZES_M[4], rexp)
        if e > 3.0:
            d = np.min([np.mean(np.linalg.norm(oc[4][list(p)] - pc[4], axis=1))
                        for p in __import__("itertools").permutations(range(4))])
            print(f"exp {exp}: subpix rot err {e:.1f}deg, corner mismatch vs pupil {d:.2f}px")
            print(f"  ocv: {np.round(oc[4], 1).tolist()}")
            print(f"  pup: {np.round(pc[4], 1).tolist()}")
