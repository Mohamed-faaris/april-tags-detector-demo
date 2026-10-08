# AprilTag Relative Pose Viewer

Detect AprilTags from a webcam or video and show each detected tag in a left-side table with columns `ID, d, x, y, z, rx, ry, rz`, relative to the **lowest visible tag ID**. `d` is Euclidean distance from the origin; translation and distance are in metres and rotation is in degrees. The origin is selected again every frame, so it changes if the lowest-ID tag leaves view.

## Run

```sh
uv sync
uv run april-tag-viewer
```

The default camera is device ID `1`. If camera 1 is unavailable or stops returning frames, the app automatically tries camera 0.

Choose a camera index, video, or image as the positional source:

```sh
uv run april-tag-viewer 1
uv run april-tag-viewer ./tags.mp4 --tag-size 0.08
uv run april-tag-viewer ./frame.png --tag-size 0.08
```

`--tag-size` is the physical edge length of the printed tag in metres. Accurate metric poses require the correct tag size and camera calibration. Without calibration, the viewer estimates focal length from the frame width and `--fov` (60° by default), which is useful for visualization but less accurate. A calibration file is an NPZ containing arrays named `camera_matrix` and `dist_coeffs`:

```sh
uv run april-tag-viewer --calibration camera.npz --tag-size 0.08
```

Override the edge length for individual IDs by repeating `--marker-size ID=METRES`; those sizes are used in pose estimation and relative position calculations:

```sh
uv run april-tag-viewer --tag-size 0.05 --marker-size 3=0.08 --marker-size 7=0.04
```

For a fixed setup, you can instead edit `MARKER_SIZES_M` near the top of `src/april_tag_relative_viewer/main.py`, for example `MARKER_SIZES_M = {3: 0.08, 7: 0.04}`. Command-line entries override code entries, and `--tag-size` is used for all IDs without a specific size.

The default family is AprilTag `36h11`, which detects the markers shown in the supplied image (IDs 0 and 1). The other supported families are `16h5` and `25h9`. Make sure the selected family matches the family used to generate your printed tags. Detection uses `pupil-apriltags`; OpenCV handles camera capture and display.

## Shortcuts

| Key | Toggle |
| --- | --- |
| `B` | Detected box boundaries |
| `I` | Top-right tag ID/origin indicator |
| `P` | Left-side relative pose table (`x, y, z, rx, ry, rz`) |
| `A` | 3D coordinate axes on each tag |
| `H` | Shortcut help overlay |
| `Q` or `Esc` | Quit |

The relative transform uses the camera-space pose of the origin tag as its reference: `R_rel = R_originᵀ R_tag`, `t_rel = R_originᵀ (t_tag - t_origin)`. Euler angles use the decomposition `Rz(rz) Ry(ry) Rx(rx)`.

## Reproducible Blender pose sanity check

Run the saved script to generate AprilTag 36h11 markers, build and render a Blender scene with known tag sizes and poses, then compare `pupil-apriltags` estimates to Blender ground truth:

```sh
uv run python scripts/blender_pose_sanity.py --blender blender
```

Outputs are saved under `artifacts/blender_pose_sanity/`: the Blender scene script, `.blend` scene, rendered PNG, source marker PNGs, and `pose_report.json`. The current Blender 5.2.2 run passed: maximum per-tag position error was 2.14 cm; relative translation error was 1.34 cm, distance error 1.17 cm, and rotation error 1.96°. The check uses ideal camera intrinsics with no lens distortion, so it is a pipeline sanity check rather than a substitute for real camera calibration.
