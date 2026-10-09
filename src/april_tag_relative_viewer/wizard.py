"""Interactive setup wizard for the AprilTag viewer (stdlib only, no rich needed).

Flow:
  1. Pick a video device from enumerated V4L2 nodes (with live preview check).
  2. Pick a calibration file (existing calibrator outputs are offered).
  3. Pick stream profile / tag size / dictionary (previous answers prefilled).
  4. Saves answers to ~/.config/april-tag-viewer/last.json for fast redo.
  5. Prints the exact `uv run april-tag-viewer ...` command that reproduces
     the session, so next time the whole wizard can be skipped.

Run:  uv run april-tag-viewer --interactive
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
from pathlib import Path

import cv2

CONFIG_PATH = Path.home() / ".config" / "april-tag-viewer" / "last.json"
CALIBRATOR_OUTPUT_GLOB = "/home/mfk01/Projects/econ-systems/tools/calibrator/output/*/camera_calibration.npz"


def load_last() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def save_last(values: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(values, indent=2) + "\n")


def ask_numbered(title: str, options: list[str], default: int = 0) -> int:
    print(f"\n{title}")
    for i, label in enumerate(options):
        mark = "  <-- last used" if i == default else ""
        print(f"  [{i + 1}] {label}{mark}")
    while True:
        raw = input(f"Choose [1-{len(options)}] (ENTER = {default + 1}): ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("Invalid choice, try again.")


def ask_value(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [ENTER = {default}]: ").strip()
    return raw if raw else default


def list_devices() -> list[dict]:
    """Enumerate V4L2 video nodes with human names (v4l2-ctl, fallback: glob)."""
    devices: list[dict] = []
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True,
                             text=True, timeout=10).stdout
        name, nodes = "", []
        for line in out.splitlines():
            if line and not line.startswith(("\t", " ")):
                if nodes and name:
                    devices.append({"name": name, "nodes": nodes})
                name, nodes = line.rstrip(":"), []
            else:
                for token in line.split():
                    if token.startswith("/dev/video"):
                        nodes.append(token)
        if nodes and name:
            devices.append({"name": name, "nodes": nodes})
    except Exception:
        pass
    if not devices:
        for node in sorted(glob.glob("/dev/video*")):
            devices.append({"name": "camera", "nodes": [node]})
    flat = []
    for dev in devices:
        for node in dev["nodes"]:
            try:
                index = int(re.search(r"video(\d+)", node).group(1))
            except Exception:
                continue
            flat.append({"index": index, "node": node, "name": dev["name"]})
    return sorted(flat, key=lambda d: d["index"])


COLOR_FOURCCS = {"MJPG", "YUYV", "YU12", "YV12", "RGB3", "BGR3", "YUV420", "NV12"}


def node_formats(node: str) -> set[str]:
    """FourCCs offered by a V4L2 node (empty set if unqueryable)."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats-ext"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return set()
    return set(re.findall(r"'(\w+)\s*'", out))


def list_profiles(node: str) -> list[dict]:
    """Parse v4l2-ctl --list-formats-ext into [{format, width, height, fps}]."""
    profiles: list[dict] = []
    try:
        out = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats-ext"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return profiles
    fmt, w, h = "YUYV", 0, 0
    for line in out.splitlines():
        m = re.search(r"\['?(\w+)'?\]", line)
        if "Size:" not in line and m and "Pixel" in line or re.match(r"\s*\[\d+\]:", line):
            m2 = re.search(r"'(\w+)'", line)
            if m2:
                fmt = m2.group(1)
            continue
        m = re.search(r"Size:\s*Discrete\s*(\d+)x(\d+)", line)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            continue
        m = re.search(r"Interval:\s*Discrete\s*[\d.]+s\s*\(([\d.]+)\s*fps\)", line)
        if m and w:
            profiles.append({"format": fmt, "width": w, "height": h, "fps": float(m.group(1))})
    # One entry per (format, size) keeping the highest fps; biggest first.
    best: dict[tuple, dict] = {}
    for p in profiles:
        key = (p["format"], p["width"], p["height"])
        if key not in best or p["fps"] > best[key]["fps"]:
            best[key] = p
    ordered = sorted(best.values(), key=lambda p: (p["width"] * p["height"], p["fps"]), reverse=True)
    mjpeg = [p for p in ordered if p["format"] == "MJPG"]
    return (mjpeg + [p for p in ordered if p["format"] != "MJPG"])[:10]


def stream_type_and_resolutions(node: str) -> tuple[str, list[str], bool]:
    """Classify a node: (stream_type, top_resolution_strings, is_color)."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats-ext"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return "Unknown", [], False
    formats = set(re.findall(r"'(\w+)\s*'", out))
    if not formats:
        return "Metadata / control", [], False
    if "Z16" in formats:
        kind, color = "Depth", False
    elif formats <= {"GREY", "Y8I", "Y12I", "UYVY", "YUYV"} and "YUYV" not in formats and "MJPG" not in formats:
        kind, color = ("Infrared / Greyscale", False)
    elif formats & COLOR_FOURCCS:
        kind, color = ("RGB / Color", True)
    else:
        kind, color = ("/".join(sorted(formats)), False)
    sizes: dict[tuple[int, int], None] = {}
    for match in re.finditer(r"Size:\s*Discrete\s*(\d+)x(\d+)", out):
        sizes[(int(match.group(1)), int(match.group(2)))] = None
    ordered = sorted(sizes, key=lambda s: s[0] * s[1], reverse=True)
    top = [f"{w}×{h}" for w, h in ordered[:3]]
    if len(ordered) > 3:
        top.append(f"(+{len(ordered) - 3} more)")
    return kind, top, color


def print_stream_table(rows: list[dict]) -> None:
    headers = ("#", "Device Node", "Camera Card", "Stream Type", "Top Resolutions")
    table = []
    for i, row in enumerate(rows, 1):
        table.append((str(i), row["node"], row["card"][:46],
                      row["kind"], ", ".join(row["resolutions"]) or "—"))
    widths = []
    for c, header in enumerate(headers):
        widths.append(max(len(header), max((len(row[c]) for row in table), default=0)))
    border = "┏" + "┳".join("━" * (w + 2) for w in widths) + "┓"
    sep = "┡" + "╇".join("━" * (w + 2) for w in widths) + "┩"
    print(border)
    print("┃ " + " ┃ ".join(h.ljust(widths[c]) for c, h in enumerate(headers)) + " ┃")
    print(sep)
    for r in table:
        print("│ " + " │ ".join(v.ljust(widths[c]) for c, v in enumerate(r)) + " │")
    print("┗" + "┻".join("━" * (w + 2) for w in widths) + "┛")


def preview_device(index: int, width: int, height: int) -> bool:
    """Open the node, warm up, show one frame for visual confirmation."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  !! {index} would not open via V4L2.")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    for _ in range(10):
        ok2, f2 = cap.read()
        if ok2:
            ok, frame = ok2, f2
    cap.release()
    if not ok:
        print("  !! opened but returned no frames.")
        return False
    print(f"  preview: {frame.shape[1]}x{frame.shape[0]} (ENTER/click-friendly check below)")
    try:
        cv2.imshow("wizard preview (any key = accept, ESC = reject)", frame)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        return key != 27
    except Exception:
        print("  (headless: preview skipped, trusting open+frames)")
        return True


def find_calibrations() -> list[str]:
    found = sorted(glob.glob(CALIBRATOR_OUTPUT_GLOB))
    extra = sorted(glob.glob(str(Path.cwd() / "*.npz")))
    return found + [e for e in extra if e not in found]


def format_command(values: dict) -> str:
    parts = ["uv run april-tag-viewer"]
    if values.get("calibration"):
        parts.append(f"--calibration {values['calibration']}")
    parts.append(f"--width {values['width']} --height {values['height']} --fps {values['fps']}")
    if values.get("fourcc"):
        parts.append(f"--fourcc {values['fourcc']}")
    parts.append(f"--tag-size {values['tag_size']}")
    if values.get("dictionary", "36h11") != "36h11":
        parts.append(f"--dictionary {values['dictionary']}")
    parts.append(str(values["source"]))
    return " ".join(parts)


def run_wizard(current: dict) -> tuple[dict, str]:
    """Camera-only wizard: show the stream table, pick a color device, preview.

    Everything else comes from last-used answers (or built-in defaults):
    top profile of the chosen node, saved calibration, tag size, family.
    Persists answers and returns (values, runnable command).
    """
    last = load_last()
    print("=== AprilTag viewer setup (camera only, rest = last used / defaults) ===")

    devices = list_devices()
    if not devices:
        raise SystemExit("No /dev/video* devices found.")
    rows: list[dict] = []
    for dev in devices:
        kind, resolutions, is_color = stream_type_and_resolutions(dev["node"])
        rows.append({"index": dev["index"], "node": dev["node"], "card": dev["name"],
                     "kind": kind, "resolutions": resolutions, "is_color": is_color})
    print_stream_table(rows)
    usable = [r for r in rows if r["is_color"]]
    if not usable:
        raise SystemExit("No color-capable video devices found.")
    default_dev = 0
    if str(last.get("source", "")).isdigit():
        for i, row in enumerate(usable):
            if row["index"] == int(last["source"]):
                default_dev = i
    choice = ask_numbered("Which video device? (depth/IR/metadata rows are not selectable)",
                          [f"index {r['index']} ({r['node']}) — {r['card']}"
                           + (f" — {', '.join(r['resolutions'])}" if r["resolutions"] else "")
                           for r in usable],
                          default_dev)
    dev = usable[choice]

    profiles = list_profiles(dev["node"])
    if profiles:
        prof = profiles[0]  # biggest resolution first; fps as offered
        width, height, fps, fourcc = prof["width"], prof["height"], prof["fps"], prof["format"]
        print(f"  profile: {width}x{height} @ {fps:.0f}fps ({fourcc})")
    else:
        print("  (profile list unavailable; using 1280x720@30 MJPG)")
        width, height, fps, fourcc = 1280, 720, 30.0, "MJPG"

    if not preview_device(dev["index"], width, height):
        print("Device rejected; restart the wizard to pick another.")
        raise SystemExit(1)

    cals = find_calibrations()
    calibration = last.get("calibration", "")
    if calibration and calibration not in cals:
        print(f"  (saved calibration missing, ignoring: {calibration})")
        calibration = ""
    tag_size = float(last.get("tag_size", current.get("tag_size", 0.05)))
    dictionary = last.get("dictionary", current.get("dictionary", "36h11"))
    print(f"  calibration: {calibration or '(none — FOV fallback)'}")
    print(f"  tag_size: {tag_size}  family: {dictionary}")

    values = {"source": dev["index"], "calibration": calibration, "width": width,
              "height": height, "fps": fps, "fourcc": fourcc,
              "tag_size": tag_size, "dictionary": dictionary}
    save_last(values)
    command = format_command(values)
    print(f"\nSaved choices to {CONFIG_PATH}.")
    print("Skip this wizard next time with:\n")
    print(f"  {command}\n")
    return values, command
