"""
Gradio demo for the ResNet18 Christmas image classifier.

Run locally with:
    pip install gradio torch torchvision pillow
    python app.py

Then open the local URL it prints (usually http://127.0.0.1:7860).

Expects `model.pkl` (your trained weights) to be in the same folder as this script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import gradio as gr

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    "christmas_cookies", "christmas_presents", "christmas_tree", "fireworks",
    "penguin", "reindeer", "santa", "snowman"
]  # alphabetical order — matches sorted(os.listdir(train_path)) label mapping in data.py


class Network(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        # weights=None: we load our own trained weights below, no need to download ImageNet weights
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model(weights_path: str = "model.pkl") -> Network:
    model = Network(num_classes=8)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

model = load_model("model.pkl")


def predict(image):
    if image is None:
        return {}
    image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    # Gradio's Label output wants a dict of {class_name: probability}
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Christmas-themed image"),
    outputs=gr.Label(num_top_classes=8, label="Predicted class"),
    title="Christmas Image Classifier (ResNet18, fine-tuned)",
    description=(
        "Fine-tuned ResNet18 achieving 90.62% accuracy on a held-out test set. "
        "Upload an image (any resolution) from one of the 8 categories: "
        + ", ".join(CLASS_NAMES) + "."
    ),
    examples=None,  # add local example image paths here if you want click-to-try samples
)

if __name__ == "__main__":
    demo.launch()
