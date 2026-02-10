
import streamlit as st
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# ---------------- CONFIG ----------------
MODEL_PATH = "multilabel_xray_model.pth"
IMAGE_SIZE = 224

DISEASE_CLASSES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Effusion",
    "Cardiomegaly"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        len(DISEASE_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# ---------------- UI ----------------
st.title("🩻 Chest X-ray Multi-Label Disease Detection")
st.write("Upload a chest X-ray image to see AI predictions.")

uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=300)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    st.subheader("🧠 Model Predictions")

    for disease, prob in zip(DISEASE_CLASSES, probs):
        percent = prob * 100
        if prob >= 0.5:
            st.success(f"{disease}: {percent:.2f}%")
        else:
            st.write(f"{disease}: {percent:.2f}%")
