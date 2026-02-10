import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
DATA_DIR = "data/ACDC_training_slices"
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4
NUM_CLASSES = 4
MODEL_PATH = "cardiac_unet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ===============================
# DATASET (.h5)
# ===============================
class CardiacMRIDataset(Dataset):
    def __init__(self, data_dir, target_size=256):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])

        with h5py.File(file_path, "r") as f:
            image = f["image"][:]   # (H, W)
            mask  = f["label"][:]   # (H, W)

        # image → (1, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # mask → (H, W)
        mask = torch.tensor(mask, dtype=torch.long)

        # resize image
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # resize mask (nearest, no channel)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(self.target_size, self.target_size),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        return image, mask

# ===============================
# DATA LOADER
# ===============================
dataset = CardiacMRIDataset(DATA_DIR, IMAGE_SIZE)
print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===============================
# U-NET MODEL
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
# TRAINING
# ===============================
model = UNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Training started...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss / len(loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("🎉 Training complete. Model saved:", MODEL_PATH)
