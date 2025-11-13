Here is a **copy-paste ready README.md**, written in a clear, simple, slightly non-native-speaker tone (but still professional and technically solid).
You can adjust links or project title anytime.

---

# Pedestrian Crossing Intent Prediction from Dashcam Video

*A prototype system for estimating pedestrian crossing intent using YOLOv8, tracking, motion cues, and road-position heuristics.*

## 📌 Overview

This project is a **dashcam-based pedestrian intent prediction system**.
It detects and tracks pedestrians from video, analyzes their motion over time, and produces an **intent score (0–1)** with three interpretable states:

* **SAFE** – normal walking, no crossing indication
* **APPROACHING** – some signs of approaching the vehicle lane
* **CROSSING** – strong indication of entering the lane (alert condition)

The goal is to build an early-warning component for ADAS-style applications and support research in transportation safety and pedestrian behavior.

The system uses:

* **YOLOv8** for detection
* **BYTETrack** for tracking
* **Temporal motion features** from bounding box history
* **Ego-motion correction**
* **Lane-band proximity**
* **Heuristic fusion** to produce an intent score
* Optional: **DeepLab road segmentation** (for road-sidewalk context)

This is **not a final ADAS system**, but a functional prototype for research, demos, and further development.

---


## 📁 Project Structure

```
dashcam-intent/
│
├── src/
│   ├── det_track.py          # YOLOv8 + BYTETrack tracking
│   ├── intent_infer.py       # motion features + intent scoring model
│   ├── overlay.py            # drawing boxes, intent text, HUD, road mask
│   ├── pose_estimator.py     # optional YOLO pose features
│   ├── road_seg_deeplab.py   # optional road segmentation (DeepLab)
│   ├── run.py                # simple track-only demo
│   ├── run_intent.py         # main pipeline (full system)
│   ├── utils.py              # TrackBuffer, cropping, helpers
│   └── __init__.py
│
├── models/                   # PIE pretrained context model (optional)
├── backup/                   # older code snapshots
├── .gitignore
└── README.md  (this file)
```

---

##  Features

### YOLOv8 + BYTETrack pedestrian tracking

Fast and stable tracking at ~10–20 FPS depending on hardware.

### Intent Scoring Model

Uses several temporal features:

* lateral vs forward motion
* approach toward lane center
* speed magnitude
* near-curb / near-lane cues
* ego-motion-compensated direction
* (optional) road-on/road-off cues

Produces a **score from 0 to 1**.

###  Interpretable States

Built-in state machine:

| Score                | State            |
| -------------------- | ---------------- |
| < approach_threshold | SAFE             |
| ≥ approach_threshold | APPROACHING      |
| ≥ crossing_threshold | CROSSING (alert) |

Thresholds are adjustable through command line.

###  Visual Debug

Overlays:

* bounding boxes
* intent scores
* state labels
* FPS and threshold
* lane-band
* optional road segmentation mask

---

##  How to Run

### 1. Install environment

```
pip install -r requirements.txt
```


### 2. Run the full intent pipeline

```
python -m src.run_intent \
    --video "sample.mp4" \
    --device 0 \
    --conf 0.35 \
    --approach_threshold 0.50 \
    --threshold 0.60 \
    --lane_band_width_ratio 0.35 \
    --out output.mp4
```

### Optional: enable road segmentation

Add:

```
--road_weights pytorch_model.bin
```

### Optional: hide live window

```
--noshow
```

---

## 🔬 Research Motivation

This project is part of my interest in **transportation engineering, pedestrian behavior modeling, and early-warning systems**.
The goal is to explore how far we can go with:

* monocular dashcam video,
* lightweight detection + tracking,
* interpretable geometric features,
* real-time computation,
* minimal training.

---


