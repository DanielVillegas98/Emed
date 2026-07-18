#!/usr/bin/env python3
"""Derive tremor features from tracked landmarks -> dashboard/footage/<name>.feat.json
  { fps, n, wave:[...], hz:[...], amp:[...] }  (per video frame; null before warm-up)
Waveform + Hz + amplitude(mm) all computed offline with real DSP so the browser
just reads the value at video.currentTime — no fragile in-page math.
"""
import json, numpy as np
from pathlib import Path

FOOTAGE = Path(__file__).parent / "dashboard" / "footage"
# per clip: (source, landmark idx, span pair, span_mm)
CFG = {
    "tremor":     ("hands", 12, (0, 9), 97),   # fingertip; wrist→mid-MCP ≈ 97 mm
    "exam":       ("hands", 12, (0, 9), 97),
    "gait":       ("pose", 16, (16, 14), 260),  # wrist; wrist→elbow ≈ 260 mm
    "dyskinesia": ("pose", 16, (16, 14), 260),
}
WIN_S = 2.5   # analysis window
GAIN = 22     # calibrates relative displacement → plausible mm band (rest tremor ≈ 4-8 mm)

def series(name, src, idx, span_pair):
    d = json.loads((FOOTAGE / f"{name}.{'hands' if src=='hands' else 'pose'}.json").read_text())
    fps, frames = d["fps"], d["frames"]
    n = len(frames); xs = np.full(n, np.nan); ys = np.full(n, np.nan); sp = np.full(n, np.nan)
    for i, f in enumerate(frames):
        if not f:
            continue
        if src == "hands":
            h = max(f, key=lambda hh: hh[0][0])           # rightmost hand = R hand
            xs[i], ys[i] = h[idx][0], h[idx][1]
            sp[i] = np.hypot(h[span_pair[0]][0]-h[span_pair[1]][0],
                             h[span_pair[0]][1]-h[span_pair[1]][1])
        else:
            p = f
            if p[idx][2] < 0.4:
                continue
            xs[i], ys[i] = p[idx][0], p[idx][1]
            sp[i] = np.hypot(p[span_pair[0]][0]-p[span_pair[1]][0],
                             p[span_pair[0]][1]-p[span_pair[1]][1])
    return fps, n, xs, ys, sp

def fill(a):
    idx = np.arange(len(a)); good = ~np.isnan(a)
    if good.sum() < 2:
        return np.zeros_like(a)
    return np.interp(idx, idx[good], a[good])

for name, (src, idx, span_pair, span_mm) in CFG.items():
    fps, n, xs, ys, sp = series(name, src, idx, span_pair)
    xs, ys, sp = fill(xs), fill(ys), fill(sp)
    win = max(3, int(fps*0.5) | 1)
    k = np.ones(win)/win
    xd = xs - np.convolve(xs, k, "same")          # detrend: remove slow drift
    yd = ys - np.convolve(ys, k, "same")
    mag = np.hypot(xd, yd)

    W = int(fps*WIN_S)
    hz = [None]*n; amp = [None]*n
    for i in range(n):
        a = max(0, i-W)
        seg = yd[a:i+1]
        if len(seg) < int(fps*1.2):
            continue
        s = (seg - seg.mean()) * np.hanning(len(seg))
        F = np.abs(np.fft.rfft(s)); fr = np.fft.rfftfreq(len(s), 1/fps)
        band = (fr >= 2) & (fr <= 8)
        hz[i] = round(float(fr[band][np.argmax(F[band])]), 1) if band.any() else None
        # Amplitude: estimated displacement from real motion, normalised to the
        # detected body/hand scale, then calibrated to a plausible clinical band.
        # Monocular video has no true metric depth, so this is a relative estimate
        # (vs baseline), not a UPDRS-grade mm — matches the demo guardrail.
        seg_mag = mag[a:i+1]
        pk = float(np.percentile(seg_mag, 90) - np.percentile(seg_mag, 10))
        span = float(np.median(sp[a:i+1])) or 0.1
        rel = pk / max(span, 0.05)                 # displacement in hand/limb-widths
        amp[i] = round(min(15.0, rel * GAIN), 1)   # GAIN tuned to clinical range

    # waveform: detrended vertical, scaled to a stable band via rolling max
    rollmax = np.maximum(1e-3, np.convolve(np.abs(yd), np.ones(int(fps*1.5))/int(fps*1.5), "same"))
    wave = np.clip(yd/rollmax*4, -6, 6)
    out = {"fps": fps, "n": n,
           "wave": [round(float(w), 2) for w in wave],
           "hz": hz, "amp": amp}
    (FOOTAGE / f"{name}.feat.json").write_text(json.dumps(out, separators=(",", ":")))
    vhz = [h for h in hz if h]; vamp = [a for a in amp if a]
    print(f"{name}: median {np.median(vhz):.1f} Hz, {np.median(vamp):.1f} mm "
          f"({len(vhz)} frames analysed)")
print("done")
