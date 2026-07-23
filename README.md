# FootVision ⚽



https://github.com/user-attachments/assets/b5755623-c500-4340-b951-4be11ca294d8



AI-powered football shot technique analysis. Upload a clip, get skeleton tracking, biomechanics measurements, and a technique score with actionable feedback.

## What it does

FootVision watches a short video of a football shot and tells you:
- **What shot type it is** — header, volley, free kick, instep drive, or curl finesse
- **How your body moved** — joint angles at the moment of contact (knee bend, torso lean)
- **How good the technique was** — a 0-10 score compared against coaching-derived ideal ranges, with specific feedback on what to fix

All of this from a single uploaded video clip — no manual annotation, no wearable sensors, just pose estimation and a trained classifier.

## Demo

Upload a clip → get back a skeleton-overlay video + technique breakdown, right in your browser.

*(screenshot / demo link here)*

## How it works

```
Video Input
   │
   ▼
YOLOv8-Pose extracts 17 body keypoints per frame
   │
   ▼
Biomechanics engine calculates joint angles + detects the contact frame
   │
   ▼
Sequence is normalized (position + scale invariant) and fed to a trained CNN
   │
   ▼
CNN classifies the shot type
   │
   ▼
Rule-based scorer compares angles to ideal ranges → score + feedback
   │
   ▼
Gradio app returns the skeleton-overlay video + full analysis
```

## Tech stack

| Component | Library |
|---|---|
| Pose estimation | Ultralytics YOLOv8-Pose (pretrained on COCO) |
| Ball detection (person selection) | YOLOv8 (object detection) |
| Video/image processing | OpenCV |
| Shot classifier | PyTorch — custom 1D CNN, trained from scratch |
| Data handling | pandas, scikit-learn, numpy |
| Web interface | Gradio |
| Visualization (dev/notebooks) | matplotlib, seaborn, Plotly |

## Project structure

```
footvision/
├── src/
│   ├── components/           # Training pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── pose_extractor.py     # YOLOv8-Pose + ball-proximity person selection
│   ├── geometry.py           # Angle calculations, contact-frame detection
│   ├── renderer.py           # Skeleton overlay drawing
│   ├── scorer.py             # Rule-based technique scoring
│   ├── logger.py / exception.py / utils.py
├── scripts/
│   └── build_dataset.py      # Raw clips -> keypoints + labels.csv
├── config/
│   ├── config.yaml
│   └── rubrics.json          # Ideal angle ranges per shot class
├── notebooks/                # EDA, pose visualization, angle analysis, training curves
├── app.py                    # Gradio application (deployment entry point)
├── requirements.txt
└── models/best_model.pth     # Trained classifier weights
```

## Dataset

- **5 shot classes:** header, volley, free kick, instep drive, curl finesse
- ~90 clips, self-collected and hand-labeled from single-player technique/tutorial footage
- Clips converted to normalized 60-frame keypoint sequences (60, 34) before training

## Model

A lightweight 1D CNN operating on normalized joint-coordinate sequences (not raw pixels):

```
Conv1d(34, 64) → ReLU → MaxPool
Conv1d(64, 128) → ReLU → MaxPool
Flatten → Linear(256) → Dropout(0.5) → Linear(num_classes)
```

**Best validation accuracy: ~80%** on a small held-out test set (5 classes). Header and instep drive are reliably distinguished; free kick, curl finesse, and volley show some overlap — expected given their biomechanical similarity and the limited dataset size.

## Setup

```bash
git clone <your-repo-url>
cd footvision
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Run the full training pipeline** (ingestion → transformation → training → evaluation):
```bash
python -m src.pipeline.training_pipeline
```

**Launch the app locally:**
```bash
python app.py
```

**Add a new shot class:**
1. Add clips to `data/raw_clips/<new_class>/`
2. Add ideal angle ranges to `config/rubrics.json`
3. Add the class name to `SHOT_CLASSES` in `src/components/data_transformation.py`
4. Update `NUM_CLASSES` in `src/components/model_trainer.py`
5. Re-run `scripts/build_dataset.py` then `training_pipeline.py`

## Limitations

- Trained on a small, self-collected dataset — a strong prototype, not production-grade accuracy
- Designed for **single-player clips** with a clear side-on angle; not reliable on match footage with multiple players in frame
- Rubric ideal-ranges are best-effort coaching estimates, not lab-verified biomechanics data
- Currently classifies 5 shot types; penalty was removed after testing due to high confusion with free kick

## Roadmap

- [ ] Reference pose comparison (DTW) against ideal technique clips
- [ ] More training data per class
- [ ] Person-tracking across frames to handle multi-player footage

## License

MIT
