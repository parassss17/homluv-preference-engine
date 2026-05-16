"""
HomLuv preference prototype -- Streamlit demo.

Loads only the small precomputed artifacts (no PyTorch here), so it
deploys on free Streamlit / Hugging Face Spaces.

Flow: user hearts/dislikes home photos -> we average the liked photos
into a taste vector -> recommend similar homes -> suggest the builder
whose homes match best, within the buyer's budget.
"""

import os
import streamlit as st

from recommender import (
    load_artifacts, build_taste_profile, recommend,
    filter_by_budget, match_builder,
)

IMG_DIR = "artifacts/images_sample"

st.set_page_config(page_title="HomLuv Preference Engine", layout="wide")


@st.cache_data
def get_data():
    return load_artifacts()


embeddings, listings = get_data()

# --- session memory: what the user has liked / disliked / seen ---
for key in ("liked", "disliked", "seen"):
    if key not in st.session_state:
        st.session_state[key] = []


def rate(idx, liked):
    st.session_state.seen.append(idx)
    (st.session_state.liked if liked else st.session_state.disliked).append(idx)


# --- header ---
st.title("HomLuv -- Visual Home Preference Engine")
st.caption(
    "A prototype of HomLuv's core idea: like the homes whose style you "
    "want, and we learn your taste and match you to a builder. "
    "Built on a public sample dataset."
)

# --- sidebar: budget + reset ---
st.sidebar.header("Buyer filters")
max_price = st.sidebar.slider(
    "Max budget ($)",
    int(listings["price"].min()),
    int(listings["price"].max()),
    int(listings["price"].median()),
    step=10000,
)
if st.sidebar.button("Reset my choices"):
    st.session_state.liked = []
    st.session_state.disliked = []
    st.session_state.seen = []
    st.rerun()

st.sidebar.write(f"Liked: {len(st.session_state.liked)}")
st.sidebar.write(f"Disliked: {len(st.session_state.disliked)}")


def show_home(row, key_prefix, with_buttons=True):
    img_path = os.path.join(IMG_DIR, row["image"])
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    st.write(f"**${int(row['price']):,}** · {int(row['bedrooms'])} bd / "
             f"{int(row['bathrooms'])} ba · zip {int(row['zipcode'])}")
    if with_buttons:
        c1, c2 = st.columns(2)
        idx = row.name
        c1.button("❤️ Like", key=f"{key_prefix}_like_{idx}",
                  on_click=rate, args=(idx, True))
        c2.button("✕ Dislike", key=f"{key_prefix}_dis_{idx}",
                  on_click=rate, args=(idx, False))


# --- discovery: rate a few homes ---
st.subheader("1. Tell us what you like")
pool = listings.drop(index=st.session_state.seen).head(6)
if pool.empty:
    st.info("You've rated everything in the sample. Reset to start over.")
else:
    cols = st.columns(3)
    for i, (_, row) in enumerate(pool.iterrows()):
        with cols[i % 3]:
            show_home(row, "pool")

# --- recommendations ---
st.subheader("2. Your matches")
if not st.session_state.liked:
    st.info("Like at least one home above to see recommendations.")
else:
    taste = build_taste_profile(
        embeddings, st.session_state.liked, st.session_state.disliked
    )
    affordable = filter_by_budget(listings, max_price=max_price)
    if affordable.empty:
        st.warning("No homes under that budget. Raise the slider on the left.")
    else:
        recs = recommend(
            embeddings.take(affordable.index, axis=0),
            affordable,
            taste,
            exclude_idx=st.session_state.seen,
            top_n=6,
        )

        builder, count = match_builder(recs)
        if builder:
            st.success(f"Best-matched builder for your taste & budget: "
                       f"**{builder}** ({count} of {len(recs)} top matches)")

        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                show_home(row, "rec", with_buttons=False)
                st.caption(f"match {row['match_score']:.2f} · {row['builder']}")
