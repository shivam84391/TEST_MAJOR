

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

# =====================================================
# 1. CONFIGURATION
# =====================================================
CSV_FILE = "data/labels.csv"
IMAGE_DIR = "data/images"
MODEL_SAVE_PATH = "multilabel_xray_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8
SEED = 42

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

# =====================================================
# 2. REPRODUCIBILITY
# =====================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# =====================================================
# 3. DATASET
# =====================================================
class ChestXrayMultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            self.df.iloc[idx, 1:].values.astype("float32")
        )

        return image, labels

# =====================================================
# 4. TRANSFORMS
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =====================================================
# 5. LOAD & SPLIT DATA
# =====================================================
full_dataset = ChestXrayMultiLabelDataset(
    CSV_FILE, IMAGE_DIR, transform
)

train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"📊 Total images: {len(full_dataset)}")
print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# =====================================================
# 6. MODEL
# =====================================================
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(
    model.classifier.in_features,
    len(DISEASE_CLASSES)
)
model = model.to(device)

# =====================================================
# 7. LOSS & OPTIMIZER
# =====================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================================================
# 8. METRIC (MICRO-F1)
# =====================================================
def micro_f1_score(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    return (2 * tp) / (2 * tp + fp + fn + 1e-8)

# =====================================================
# 9. TRAINING LOOP
# =====================================================
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # ---------- TRAIN ----------
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs))
            all_labels.append(labels)

    val_loss /= len(val_loader)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = micro_f1_score(all_labels, all_preds)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Micro-F1: {f1:.4f}"
    )

    # ---------- SAVE BEST MODEL ----------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(" Best model saved")

print("\n Training completed successfully")
print("📦 Final model:", MODEL_SAVE_PATH)



# 7 February 2026

# Today carried a quiet weight, the kind that stays with you even when nothing obvious goes wrong.
# I went to the stadium and ran. For the first time in a while, my body cooperated. I managed 4 km, and the recovery felt better than before. It felt like proof that even after everything, I’m still moving forward—slowly, but honestly. Later, I played badminton. Four matches, one win. Not much on paper, but being there, breathing hard, feeling alive—it mattered.
# This year began with the biggest loss of my life. I lost my father. Some days I move through life normally, and some days the absence feels unbearably loud. Today, his absence stayed quietly with me. I keep wondering if he would be proud of me for still trying, for holding myself together when everything feels fragile.
# I tried to stay focused on my major project today. I also made a conscious decision to not disturb Nikki. She should focus on her studies and get proper rest. I didn’t ask her how she has been, even though I thought about her. I had promised her I’d give her the bottle, but I couldn’t manage that either. Life got too busy, too heavy all at once. I’ll give it to her when things settle a little, when I’m more stable. I know she’ll understand—she always does.
# Still, my heart remembers things clearly. She stood by me on a day when no one else did. When I felt completely alone, she was there. That kind of presence doesn’t fade. She’s a genuinely good soul, and I always wish the best for her—hamesha acha hi ho uske saath.
# This year started with pain and uncertainty. I truly hope she gets a job soon. The day she tells me she’s selected, it will bring a small sense of relief, maybe even a smile. She works incredibly hard, has a strong fighting spirit, and learns things quickly. She never once said no to me, and that kindness stays with me.
# Tonight, I’m ending the day quietly—washing dishes and writing this journal after 20 days. Life feels fragile, but I’m still standing, still trying. Maybe that’s enough for today.
# Miss you, Papa.
# I hope I’m doing okay.



