# AprilTag Pose Sanity — Final Report

## Setup
- Blender 5.2.2 renders (640×480, fx=fy=500px, no distortion) of tag36h11 **ID 0** (0.095 m) and **ID 4** (0.06725 m); same textures reused everywhere.
- 30 experiments, different per-tag poses (`experiments/poses_30.csv`, seed 11), FOV-validated pre-render (corners inside frame @25 px margin, tilt < 55°).
- Thresholds: pos ≤ 0.03 m, rel. trans ≤ 0.02 m, rel. dist ≤ 0.02 m, rot ≤ 5°.
- Inputs frozen once (`experiments/inputs/<id>/{input.png, ground_truth.json}` + `inputs/map.json`). All configs/libraries tested on frozen inputs — no re-rendering.

## Best pipeline: `best_sig08_spnp` ✅
`quad_decimate=0.5` + `quad_sigma=0.8` + solvePnP pose backend (true tag size, correspondence disambiguated by detector rotation prior). **30/30**, rel. error **7.3 mm / 6.7 mm / 0.61°**.

| Pipeline | Pass | tag4 pos/rot | rel. trans/dist/rot |
|---|---|---|---|
| baseline (qd=1.0) | 29/30 | 21.3 mm / 0.58° | 12.9 / 11.2 mm / 0.77° |
| qd05 | 30/30 | 20.4 mm / 0.61° | 12.1 / 10.5 mm / 0.81° |
| qd05 + solvePnP | 30/30 | 20.2 mm / 0.47° | 11.8 / 10.3 mm / 0.60° |
| qd05 + σ=0.8 | 30/30 | 15.2 mm / 0.66° | 7.6 / 6.8 mm / 0.81° |
| **qd05 + σ=0.8 + solvePnP** | **30/30** | **15.1 mm / 0.51°** | **7.3 / 6.7 mm / 0.61°** |
| refine_edges=0 | 27/30 ❌ | 12.9 mm / 2.51° | 7.0 / 5.6 mm / 2.77° |

Blur σ=0.8 halves position error; solvePnP cuts rotation ~25%; edge refinement is load-bearing (off → 27/30).

## Library comparison (same inputs, same solvePnP backend, same metrics)
pupil / dt / apriltag are all **AprilTag 3** C-core wrappers (verified bundled symbols); OpenCV is the independent implementation.

| Library | Pass | Speed | tag4 pos/rot | rel. trans/rot | Verdict |
|---|---|---|---|---|---|
| pupil-apriltags | 30/30 | 4.3 ms | 15.1 mm / 0.51° | 7.3 mm / 0.61° | ✅ best overall |
| dt-apriltags | 30/30 | 4.1 ms | identical | identical | ✅ tied best, fastest of the best |
| apriltag (pip) | 28/30 | 2.8 ms | **5.1 mm** / 1.59° | 11.0 mm / 1.74° | best raw position, rotation too noisy |
| OpenCV | 13/30 | 2.0 ms | 29.5 mm / 6.18° | 19.5 mm / 6.97° | fastest, not accurate enough here |
| OpenCV + subpix | 9/30 | 2.0 ms | 32.9 mm / 5.63° | 21.2 mm / 5.65° | subpix matches pupil on near tag (0.29°) but collapses on small/far tag |

Why OpenCV lags: default integer-pixel corners (≈0.9 px quantization → ~2° errors), and on the small far tag its corners intermittently lock onto wrong edges (24–37° outliers despite agreeing with pupil to <1 px on good frames). Fixing that needs IPPE-style ambiguity handling, not just finer corners.

## Recommendation
Use **dt-apriltags** (or pupil-apriltags) with `quad_decimate=0.5, quad_sigma=0.8, refine_edges=1` + solvePnP-on-true-size pose backend.

## Reproduce (all via `uv run python`)
- Best pipeline: `scripts/blender_pose_sanity.py --poses-csv experiments/poses_30.csv --stage process --pipeline best --quad-decimate 0.5 --quad-sigma 0.8 --pose-backend solvepnp`
- Library benchmark: `scripts/lib_benchmark.py [--libs pupil dt apriltag opencv opencv_subpix]`
- Full table: `experiments/outputs/lib_compare.json`. Per-run artefacts: `experiments/outputs/<pipeline>/<id>/{annotated.png, report.json}` + `summary.{csv,json}`.
