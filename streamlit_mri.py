import streamlit as st
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "cardiac_unet.pth"
IMAGE_SIZE = 256
NUM_CLASSES = 4   # background, LV, RV, MYO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Cardiac MRI Segmentation",
    layout="wide"
)

# ===============================
# MODEL DEFINITION (SAME AS TRAINING)
# ===============================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.middle = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, NUM_CLASSES, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        m = self.middle(self.pool(c3))
        u3 = self.conv3(torch.cat([self.up3(m), c3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ===============================
# HELPER FUNCTIONS
# ===============================
def preprocess(image):
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    image = F.interpolate(
        image,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )
    return image.to(DEVICE)

def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1)
    return pred.squeeze(0).cpu().numpy()

def lv_analysis(mask):
    lv_pixels = np.sum(mask == 1)
    total_pixels = mask.size
    ratio = lv_pixels / total_pixels
    status = "Enlarged" if ratio > 0.15 else "Normal"
    return ratio, status

# ===============================
# STREAMLIT UI
# ===============================
st.title("🫀 Cardiac MRI Segmentation & LV Analysis")
st.markdown("Upload a **Cardiac MRI `.h5` file** to segment ventricles.")

uploaded_file = st.file_uploader(
    "Upload MRI (.h5)",
    type=["h5"]
)

if uploaded_file:
    with h5py.File(uploaded_file, "r") as f:
        image = f["image"][:]

    image_tensor = preprocess(image)
    mask = predict(image_tensor)

    lv_ratio, lv_status = lv_analysis(mask)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, clamp=True)

    with col2:
        st.subheader("Segmentation Overlay")
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.imshow(mask, alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("🩺 Clinical Result")
    st.write(f"**LV Area Ratio:** `{lv_ratio:.3f}`")
    if lv_status == "Enlarged":
        st.error(f"**LV Status:** {lv_status}")
    else:
        st.success(f"**LV Status:** {lv_status}")

    st.caption("⚠️ For academic and research use only.")
