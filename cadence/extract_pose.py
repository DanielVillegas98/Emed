#!/usr/bin/env python3
"""MediaPipe pose extraction v2 — full model, patient-lock, gap-fill, smoothing.

Output per clip: dashboard/footage/<name>.pose.json
  { "fps": 25.0, "n": 2100, "frames": [ [[x,y,vis]*33] | null, ... ] }
"""
import cv2, json
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

SCRATCH = "/private/tmp/claude-501/-Users-chumacbook/b8adae12-7762-43e7-ac27-a8f9eac3d437/scratchpad"
MODEL = f"{SCRATCH}/pose_landmarker_full.task"
FOOTAGE = Path(__file__).parent / "dashboard" / "footage"
CLIPS = ["exam", "gait", "tremor", "dyskinesia"]
MAX_GAP = 20          # interpolate gaps up to this many frames
EMA = 0.65            # smoothing: new*EMA + prev*(1-EMA); mild, keeps motion

def torso_brightness(gray, p, w, h):
    xs = [p[j].x for j in (11, 12, 23, 24)]; ys = [p[j].y for j in (11, 12, 23, 24)]
    cx, cy = int(sum(xs) / 4 * w), int(sum(ys) / 4 * h)
    x0, x1 = max(0, cx - 8), min(w, cx + 8); y0, y1 = max(0, cy - 8), min(h, cy + 8)
    return float(gray[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0

def pick(res, gray, w, h, exam):
    if not res.pose_landmarks:
        return None
    if not exam:
        return res.pose_landmarks[0]
    best, bb = None, 0
    for p in res.pose_landmarks:
        b = torso_brightness(gray, p, w, h)
        if b > bb:
            bb, best = b, p
    return best if bb > 105 else None      # patient wears light sweater

def interpolate(frames):
    """Fill gaps <= MAX_GAP with linear interpolation between good frames."""
    n = len(frames); i = 0; filled = 0
    while i < n:
        if frames[i] is not None:
            i += 1; continue
        j = i
        while j < n and frames[j] is None:
            j += 1
        prev, nxt = (frames[i - 1] if i > 0 else None), (frames[j] if j < n else None)
        gap = j - i
        if prev and nxt and gap <= MAX_GAP:
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                frames[i + k] = [[round(a[0] + (b[0] - a[0]) * t, 4),
                                  round(a[1] + (b[1] - a[1]) * t, 4),
                                  round(min(a[2], b[2]), 2)]
                                 for a, b in zip(prev, nxt)]
            filled += gap
        i = j
    return filled

def smooth(frames):
    prev = None
    for f in frames:
        if f is None:
            prev = None; continue
        if prev is not None:
            for k in range(33):
                f[k][0] = round(f[k][0] * EMA + prev[k][0] * (1 - EMA), 4)
                f[k][1] = round(f[k][1] * EMA + prev[k][1] * (1 - EMA), 4)
        prev = f

for name in CLIPS:
    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2 if name == "exam" else 1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3))
    cap = cv2.VideoCapture(str(FOOTAGE / f"{name}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = gray.shape
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                  int(i * 1000 / fps))
        p = pick(res, gray, w, h, name == "exam")
        frames.append([[round(l.x, 4), round(l.y, 4),
                        round(l.visibility if l.visibility is not None else 1.0, 2)]
                       for l in p] if p else None)
        i += 1
    cap.release(); lm.close()
    raw = sum(1 for f in frames if f)
    filled = interpolate(frames)
    smooth(frames)
    good = sum(1 for f in frames if f)
    (FOOTAGE / f"{name}.pose.json").write_text(
        json.dumps({"fps": fps, "n": len(frames), "frames": frames}, separators=(",", ":")))
    print(f"{name}: raw {raw}/{len(frames)} + {filled} interpolated = {good} "
          f"({100*good//len(frames)}%)", flush=True)
print("done")
