# Fine-Tuned CNN for Multi-Class Image Classification Under Real-World Constraints

A transfer-learning-based image classifier built and trained from scratch (PyTorch), tackling
variable-resolution images across 8 classes with a small training set — achieving **90.62%
accuracy on a held-out, secret test set** (evaluated independently, not on the public leaderboard).

> This project originated as a graduate coursework challenge (Deep Learning, University of Siegen)
> and is presented here as a demonstration of an end-to-end computer vision pipeline: data handling,
> transfer learning, regularization, and generalization to unseen data. The original private dataset
> and course evaluation scripts are not included — only my own code, methodology, and results.

---

## Problem

Classify images into 8 categories, where:
- **Images vary in resolution** — no guarantee of consistent input size.
- **Training data is limited** — a few hundred images per class, not a large-scale dataset.
- **Evaluation happens on a secret held-out test set**, separate from the public leaderboard used
  for iteration — meaning the model has to genuinely generalize, not just overfit to leaderboard
  feedback.

This mirrors a common real-world CV scenario: production data rarely arrives at a fixed
resolution or in ImageNet-scale quantities, and the metric that matters is performance on data the
model has never seen or been tuned against.

## Approach

**Architecture — transfer learning over training from scratch.**
Rather than train a CNN from random initialization (which struggles with limited data), I fine-tuned
a **ResNet18 pretrained on ImageNet**, replacing its final layer with a small custom classification
head:

```
ResNet18 (ImageNet-pretrained backbone)
  → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 8)
```

This choice trades a from-scratch architecture for a backbone that already understands general
visual features (edges, textures, shapes), which matters far more than model novelty when the
labeled dataset is small. It's the same reasoning used in most practical CV pipelines: don't
re-learn the wheel when transfer learning is available and appropriate.

**Handling variable input sizes.**
All images are resized to a fixed `224×224` (matching ResNet's expected input) as part of the
`Dataset` pipeline, with `ImageNet` mean/std normalization so the pretrained weights receive inputs
in the distribution they were trained on.

**Regularization and augmentation** (to fight overfitting on a small dataset):
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness/contrast/saturation ±0.2)
- Dropout (p=0.3) in the classification head

**Data pipeline correctness.**
The test-time loader enforces strict numeric ordering of unlabeled images (`0.png, 1.png, 2.png…`
rather than lexicographic, which would misorder `10.png` before `2.png`) so that predictions align
correctly with expected IDs — a small but easy-to-miss correctness bug in image pipelines.

## Model interpretability (Grad-CAM)

To verify the model is learning meaningful visual features — not just shortcuts in the
data — I generated **Grad-CAM** heatmaps on the last convolutional block of the fine-tuned
ResNet18, run on unseen validation images:

![Grad-CAM examples](results/gradcam_grid.jpg)

The activations consistently localize on the actual subject of each image (the santa figure,
the snowman's body, the penguin) rather than background or edge artifacts — a good sign the
model generalizes on genuine visual features rather than spurious correlations, which matters
more than the accuracy number alone when judging whether a model will hold up on truly new data.

## Results

| Metric | Value |
|---|---|
| Test accuracy (secret held-out set) | **90.62%** |
| Number of classes | 8 |
| Training set size | A few hundred images, split across 8 classes |
| Backbone | ResNet18 (ImageNet pretrained), fully fine-tuned (no frozen layers) |
| Epochs | 15 |

*(Grading was scaled linearly between 15% accuracy = 1 point and 90% accuracy = full points,
with 12.5% representing random chance across 8 classes — this model cleared the top band.)*

## What I'd improve next

- Compare fine-tuned ResNet18 against a frozen-backbone baseline (linear probe only) to quantify
  how much fine-tuning contributed vs. the pretrained features alone.
- Try a slightly deeper backbone (ResNet34/50) or EfficientNet to see if accuracy improves without
  overfitting, given the small dataset size.

## Repo structure

```
├── README.md
├── model.py       # Network definition (ResNet18 + custom head)
├── data.py        # Dataset class with augmentation + strict numeric test ordering
├── train.py        # Training loop
├── predict.py     # Inference script, produces Id,Category CSV
├── samples/        # A handful of example images (not the full dataset)
└── results/         # Training curves, confusion matrix, Grad-CAM examples
```

## Notes on the dataset

The original dataset was provided as part of a university course and is not redistributed here.
It consisted of 8 holiday/object categories with images at varying resolutions. If you'd like to
reproduce this pipeline, swap in any multi-class image folder structured as:

```
data/train/<class_name>/*.jpg
data/val/*.jpg   # unlabeled, for inference
```

---

**Tech stack:** Python, PyTorch, torchvision, PIL

## Try it yourself (local demo)

A small Gradio app (`app.py`) is included for interactive testing:

```bash
pip install gradio torch torchvision pillow
python app.py
```

Then open the local URL it prints and upload any image — the model outputs a probability
distribution across all 8 classes. Requires `model.pkl` (your own trained weights) in the same
folder; weights are not included in this repo since they were trained on private course data.

**Example run:**

![Demo screenshot](results/demo_screenshot.png)

*Uploaded image correctly classified as `santa` with 97% confidence; the remaining probability
mass falls on visually related classes (`reindeer`, `christmas_presents`) rather than unrelated
ones — a good sign the model's confidence is well-calibrated, not just a random high number.*
