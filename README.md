# Real-Time Emotion Detection (FER2013)

A webcam application that finds a face (OpenCV Haar cascade) and classifies its emotion in real time
with an ensemble of three PyTorch models: a custom CNN, ResNet50 and EfficientNet-B0.

Built during my internship at İTOSAM (Aug–Sep 2026).
Case study with charts and discussion: https://burakb.netlify.app/emotion-detection.html

## Results (FER2013 test set, 7,178 images)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Own CNN (5 conv blocks, 224px, MixUp) | 61% | 0.59 |
| ResNet50 (layer4 fine-tuned, MixUp) | 66% | 0.64 |
| EfficientNet-B0 (fine-tuned, MixUp) | 67% | 0.66 |
| **Ensemble (3 models, averaged softmax)** | **70%** | **0.68** |

A 90% target was set for the project. It was not reached, and published results on FER2013 top out at roughly
73–76% because of its noisy labels. See the case study for limitations.

## Techniques

Class weighting, Focal Loss, AdamW, label smoothing, MixUp, early stopping, BatchNorm, Dropout,
data augmentation, transfer learning (freezing + differential learning rates), ensemble averaging,
frame skipping for real-time inference.

## Project structure

| File | Purpose |
|---|---|
| `model.py`, `train.py` | Own CNN and its training |
| `model_resnet.py`, `train_resnet.py` | ResNet50 transfer learning |
| `model_efficientnet.py`, `train_efficientnet.py` | EfficientNet-B0 transfer learning |
| `focal_loss.py` | Focal Loss implementation |
| `evaluate_test*.py` | Per-model test evaluation |
| `3+1model_evaluate.py` | Ensemble evaluation + confusion matrix |
| `cascade.py` | Haar cascade face/eye detection (first step of the project) |
| `app.py` | Real-time PyQt5 webcam app (3-model ensemble) |
| `emotion_detection_rapor.ipynb` | Report notebook with graphs |
| `*.png`, `*.txt` | Loss curves, confusion matrix, result files |

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or the CPU build
pip install opencv-python==4.10.0.84 PyQt5 numpy scikit-learn matplotlib pillow
```

`opencv-python==4.10.0.84` is pinned on purpose: version 5.0.0 had a `cv2.imshow` / `CascadeClassifier` bug on my machine.

## Data

Download FER2013 from Kaggle (https://www.kaggle.com/datasets/msambare/fer2013), extract it, and put the
`train` and `test` folders where the scripts expect them (check the dataset path at the top of `train.py`).
The dataset is not included in this repository.

## Trained models

Checkpoints are too large for the repository. Download the three final models from the
[Releases](../../releases) page and put them next to `app.py`:

- `model_FINAL_cnn_224px_mixup.pth`
- `model_FINAL_resnet50_mixup.pth`
- `model_FINAL_efficientnet_mixup.pth`

## Run

```bash
python app.py
```

## Limitations

The model predicts a facial-expression label, not a real inner emotional state, and it should not be used to make
decisions about people. Accuracy was measured on FER2013 only; webcam performance in other conditions was checked by eye.
