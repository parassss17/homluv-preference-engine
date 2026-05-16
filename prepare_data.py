"""
Run this ONCE on your own machine after downloading the dataset
(see data/README.md). It does the heavy work so the deployed app
doesn't have to:

  1. read HousesInfo.txt          -> a clean listings table
  2. images -> number vectors     using a PRETRAINED model (not trained by us)
  3. group homes into "builders"  by style + price (KMeans)
  4. save small artifacts/        which is all the app needs

Usage:
    pip install -r requirements-dev.txt
    python prepare_data.py
"""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import models, transforms

DATA_DIR = "data/Houses-dataset"
ART_DIR = "artifacts"
IMG_OUT = os.path.join(ART_DIR, "images_sample")
ROOMS = ["bedroom", "bathroom", "kitchen", "frontal"]
N_BUILDERS = 8


def load_listings():
    """HousesInfo.txt -> DataFrame. Columns are space-separated."""
    path = os.path.join(DATA_DIR, "HousesInfo.txt")
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["bedrooms", "bathrooms", "area", "zipcode", "price"],
    )
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def build_feature_extractor():
    """A pretrained MobileNetV2 with its final classifier removed.

    We are NOT training this. It was trained on ImageNet by others; we
    only push images through it to get a 1280-number 'style' vector.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


# Standard ImageNet preprocessing expected by the pretrained model.
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def image_vector(model, path):
    """One image -> one 1280-length vector."""
    img = Image.open(path).convert("RGB")
    batch = PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        return model(batch).squeeze(0).numpy()


def main():
    listings = load_listings()
    model = build_feature_extractor()

    os.makedirs(IMG_OUT, exist_ok=True)
    vectors = []

    for house_id in listings["id"]:
        # Average the 4 room photos into one style vector for the home.
        per_room = []
        for room in ROOMS:
            p = os.path.join(DATA_DIR, f"{house_id}_{room}.jpg")
            if os.path.exists(p):
                per_room.append(image_vector(model, p))
        vectors.append(np.mean(per_room, axis=0))

        # Save a downsized frontal photo for the app to display.
        frontal = os.path.join(DATA_DIR, f"{house_id}_frontal.jpg")
        if os.path.exists(frontal):
            im = Image.open(frontal).convert("RGB")
            im.thumbnail((400, 400))
            im.save(os.path.join(IMG_OUT, f"{house_id}.jpg"), quality=80)

        if house_id % 50 == 0:
            print(f"  processed {house_id} homes...")

    embeddings = np.vstack(vectors).astype(np.float32)

    # "Builders" don't exist in this public sample, so we simulate them:
    # group homes that look alike and cost alike (style + price).
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    price_scaled = StandardScaler().fit_transform(listings[["price"]])
    cluster_input = np.hstack([embeddings, price_scaled])
    labels = KMeans(n_clusters=N_BUILDERS, random_state=42, n_init=10).fit_predict(cluster_input)
    listings["builder"] = [f"Builder {l + 1:02d}" for l in labels]
    listings["image"] = listings["id"].astype(str) + ".jpg"

    os.makedirs(ART_DIR, exist_ok=True)
    np.save(os.path.join(ART_DIR, "embeddings.npy"), embeddings)
    listings.to_csv(os.path.join(ART_DIR, "listings.csv"), index=False)

    print(f"\nDone. {len(listings)} homes, {embeddings.shape[1]}-dim vectors.")
    print(f"Wrote: {ART_DIR}/embeddings.npy, {ART_DIR}/listings.csv, {IMG_OUT}/*.jpg")


if __name__ == "__main__":
    main()
