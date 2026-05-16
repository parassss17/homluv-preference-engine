---
title: HomLuv Preference Engine
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# HomLuv — Visual Home Preference Engine (Prototype)

A working prototype of [HomLuv](https://www.homluv.com/)'s core idea, built
during an industrial internship at **EX Squared Solutions** (a Builders
Digital Experience / BDX company).

HomLuv is a visual search & discovery platform for new-home buyers: instead of
typing filters, buyers **like / dislike home photos**, and the platform learns
their style and matches them to the right **homebuilder** by price and
location. This repo reproduces that loop end-to-end on a small **public sample
dataset**, so the technique is demonstrable without any private data.

> **Scope note (read this):** This is a prototype on the public *Houses
> Dataset*, not HomLuv's production system. The dataset has no builder field,
> so builder portfolios here are **simulated** by clustering homes on visual
> style + price. All numbers in the notebooks come from this code on this
> sample — nothing is taken from the company.

---

## The idea in one line

Average the image vectors of the homes a buyer *liked* into a single "taste"
vector, then rank every home by how close it is to that taste — and suggest
the builder whose homes match best, within budget.

---

## Pipeline

```
data/Houses-dataset  (535 homes: 4 photos each + price/zip)
        |
        v
prepare_data.py   (run ONCE, locally)
   - photos --> 1280-dim style vectors   [pretrained MobileNetV2, NOT trained by us]
   - KMeans on style+price --> simulated "builders"
        |
        v
artifacts/   embeddings.npy  +  listings.csv  +  images_sample/   <-- small, committed
        |
        +--> notebooks/01_explore   (EDA)
        +--> notebooks/02_model     (recommender demo + classifier accuracy)
        |
        v
app.py   Streamlit demo: heart photos --> taste vector --> matches + builder
```

The heavy image model runs **only** in `prepare_data.py` on your machine.
The deployed app loads just the small cached vectors (NumPy / scikit-learn),
so it runs on free Streamlit Community Cloud or Hugging Face Spaces.

---

## What each library does

| Step | Library | Role |
|---|---|---|
| Parse `HousesInfo.txt`, tables | pandas | clean the listings |
| Load / resize photos | Pillow | image I/O |
| Photo → number vector (once) | PyTorch / torchvision | pretrained feature extractor |
| Cache vectors | NumPy | `embeddings.npy` |
| Taste profile + ranking | NumPy | cosine similarity |
| Simulated builders | scikit-learn | KMeans clustering |
| Like/dislike classifier | scikit-learn | LogisticRegression |
| Demo UI | Streamlit | the clickable app |

---

## Run it locally

```bash
# 1. get the data (see data/README.md), then:
pip install -r requirements-dev.txt
python prepare_data.py          # one-time: builds artifacts/

# 2. explore / verify
jupyter notebook notebooks/

# 3. run the demo
streamlit run app.py
```

Deployed app needs only `requirements.txt` (no PyTorch).

---

## Author

**Paras Beniwal** — built during the EX Squared Solutions internship
(Jun–Aug 2025), on the HomLuv project.
