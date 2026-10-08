# AprilTag Pose Sanity — Brief Report

## Setup
- Blender 5.2.2 renders (640×480, fx=fy=500px, no distortion) of tag36h11 tags **ID 0** (0.095 m) and **ID 4** (0.06725 m); same tag textures reused for all runs.
- 30 experiments with different per-tag positions/orientations (`experiments/poses_30.csv`, seed 11). Every pose is FOV-validated before rendering: all corners inside the image (25 px margin), tilt < 55°, centres ≥ 90 px apart.
- Pose estimated with pupil-apriltags; thresholds: pos ≤ 0.03 m, rel. trans ≤ 0.02 m, rel. dist ≤ 0.02 m, rot ≤ 5.0°.
- Inputs frozen once (`experiments/inputs/<id>/{input.png, ground_truth.json}`); pipelines re-run on frozen inputs without re-rendering (`experiments/outputs/<pipeline>/`).

## Results
| Pipeline | Pass | Mean tag0 (pos/rot) | Mean tag4 (pos/rot) | Mean rel. (trans/dist/rot) |
|---|---|---|---|---|
| baseline (quad_decimate=1.0) | **29/30** | 0.0092 m / 0.36° | 0.0213 m / 0.58° | 0.0129 m / 0.0112 m / 0.77° |
| qd05 (quad_decimate=0.5) | **30/30** | 0.0092 m / 0.36° | 0.0204 m / 0.61° | 0.0121 m / 0.0105 m / 0.81° |

- Only failure: `013` under baseline — both tags detected with small individual errors, but relative translation error 0.0203 m marginally exceeds the 0.02 m threshold. It passes under qd05.
- Finer decimation (0.5) slightly improves position/relative-translation estimates at the cost of runtime; rotation accuracy is equivalent.

## Artefacts
- `experiments/inputs/map.json` — ground-truth map for all 30 experiments.
- `experiments/outputs/<pipeline>/{<id>/annotated.png, <id>/report.json, summary.csv, summary.json}`.

## Reproduce
- Render: `--poses-csv experiments/poses_30.csv --stage render`
- New pipeline on frozen inputs: `--poses-csv experiments/poses_30.csv --stage process --pipeline <name> --quad-decimate <v>`
