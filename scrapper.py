import pandas as pd
import os
import shutil

CSV_FILE = "sample_labels.csv"
IMAGE_DIR = "images"

OUT_IMG_DIR = "data/images"
OUT_LABEL_FILE = "data/labels.csv"

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

os.makedirs(OUT_IMG_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

rows = []

for _, row in df.iterrows():
    img = row["Image Index"]
    findings = str(row["Finding Labels"])

    src = os.path.join(IMAGE_DIR, img)
    if not os.path.exists(src):
        continue

    shutil.copy(src, os.path.join(OUT_IMG_DIR, img))

    label_vector = []
    for disease in DISEASE_CLASSES:
        label_vector.append(1 if disease in findings else 0)

    rows.append([img] + label_vector)

columns = ["image"] + DISEASE_CLASSES
label_df = pd.DataFrame(rows, columns=columns)
label_df.to_csv(OUT_LABEL_FILE, index=False)

print("✅ Multi-label dataset created")
print("Images:", len(label_df))
