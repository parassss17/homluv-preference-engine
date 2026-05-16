# Dataset

This project uses the public **Houses Dataset** (Ahmed & Moustafa, 2016) —
real estate listings with interior + exterior photos and price/location metadata.

## How to download

1. Go to: https://github.com/emanhamed/Houses-dataset
2. Download the repository (Code → Download ZIP).
3. Unzip it and copy the inner `Houses Dataset` folder here, renamed to:

   ```
   data/Houses-dataset/
   ```

   After this you should have:

   ```
   data/Houses-dataset/
   ├── HousesInfo.txt          # bedrooms bathrooms area zipcode price (one row per house)
   ├── 1_bathroom.jpg
   ├── 1_bedroom.jpg
   ├── 1_frontal.jpg
   ├── 1_kitchen.jpg
   ├── 2_bathroom.jpg
   └── ...
   ```

## What it contains

- **535 houses**, each with **4 images**: bathroom, bedroom, kitchen, frontal (exterior).
- `HousesInfo.txt`: 5 space-separated columns per row —
  `bedrooms  bathrooms  area_sqft  zipcode  price`.

The raw folder is git-ignored (too large to commit). Only the small
precomputed `artifacts/` are committed, which is all the deployed app needs.

> Note: this public sample has **no builder field**. Builder portfolios in this
> project are *simulated* by clustering listings on style + price (see README).
