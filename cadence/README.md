# Cadence — video + voice capture for Parkinson's home monitoring

Closing the *snapshot gap*: a Parkinson's patient is seen ~15 minutes every 6–18
months, so dosing runs on a snapshot. Cadence turns the home into the instrument —
**camera → movement, microphone → speech** — captured continuously and turned into
one longitudinal ON/OFF record + titration flag for the clinician.

## Live demo
- Capture / edge view: https://cadence-dashboard-henrychentp.vercel.app/capture.html
- Doctor dashboard: https://cadence-dashboard-henrychentp.vercel.app/index.html
- Pitch slides: https://cadence-dashboard-henrychentp.vercel.app/slides.html

## The video + voice capture actually works — here's the proof

**Video capture (real MediaPipe pose/hand tracking).**
`extract_pose.py` runs Google MediaPipe (PoseLandmarker full model, 33 landmarks;
HandLandmarker, 21 points) over real Parkinson's patient footage and writes the
tracked skeleton per frame to `footage/<clip>.pose.json` / `.hands.json`.
`extract_features.py` then derives the tremor signal from those landmarks:
band-passed wrist/finger displacement → **FFT frequency (the 4–6 Hz rest tremor)**
and amplitude, written to `footage/<clip>.feat.json`.
In `capture.html` the tracked skeleton is drawn frame-accurately over the video and
the live motor panel reads Hz/amplitude from the real track — open it and click
**Home cam / Tremor / Dyskinesia** to see the skeleton lock onto real patients.

**Voice capture (real acoustic biomarker analysis).**
Real single-speaker recordings of a Parkinson's patient vs a healthy speaker are
analysed the same way (pitch contour, melodic range, monotone stretches, loudness).
The **Voice biomarkers** panel plays each clip and shows the patient's flat,
monotone voice (reduced melodic range) against the healthy speaker's — the earliest,
most sensitive speech sign in Parkinson's, from a passive mic with no extra hardware.

## Run locally
```bash
cd cadence
python3 serve.py 8742      # threaded + Range-capable (needed for video streaming)
# open http://localhost:8742/capture.html
```
To regenerate the tracking data from the source clips:
```bash
pip install mediapipe opencv-python numpy
python3 extract_pose.py        # video → pose/hand landmarks
python3 extract_features.py    # landmarks → tremor Hz/amplitude
```

## Files
- `capture.html` — edge/capture view: live pose skeleton on real footage + voice biomarkers
- `index.html` — clinician dashboard (ON/OFF diary, dose table, titration flag)
- `slides.html` — pitch deck
- `extract_pose.py` — MediaPipe pose/hand tracking (video capture)
- `extract_features.py` — tremor DSP from landmarks
- `footage/` — real patient clips + tracked landmark JSON + voice samples

## How this fits with the rest of Emed
Cadence is the **longitudinal / titration** layer. It uses the same MediaPipe pose
stream as the acute fall-detection pipeline (`vision_tracker.py`) and the voice
agent (`voice_triage.py`) — one capture pipeline, two time-scales: **seconds for
safety, months for dosing.**
