import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(
    page_title="ESG F1 Copilot",
    layout="wide"
)

st.title("🏎️🌱 ESG F1 Copilot")
st.caption("Corporate ESG reasoning using Formula 1 sustainability data")

# ===============================
# PATH CONFIGURATION
# ===============================
BASE_PATH = "formula-1-dataset-race-data-and-telemetry/Directories/LapData"
DATA_DIR = "data"

# ===============================
# LOAD ESG KNOWLEDGE FILES
# ===============================
documents = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            documents.append(f.read())

# ===============================
# TF-IDF VECTOR STORE
# ===============================
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(documents)

def retrieve_context(query, k=2):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return "\n".join([documents[i] for i in top_indices])

# ===============================
# LOAD LLM (STABLE)
# ===============================
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

# ===============================
# LOAD AVAILABLE RACES
# ===============================
def list_races():
    files = os.listdir(BASE_PATH)
    race_files = [f for f in files if f.endswith(".csv")]
    race_names = [f.replace("_", " ").replace(".csv", "") for f in race_files]
    return race_files, race_names

race_files, race_names = list_races()

@st.cache_data
def load_data(selected_file):
    return pd.read_csv(os.path.join(BASE_PATH, selected_file))

# ===============================
# FEATURE ENGINEERING
# ===============================
def prepare_features(df):
    df = df.copy()

    df["SectorTotal"] = (
        df["Sector1Time"] + df["Sector2Time"] + df["Sector3Time"]
    )

    df["LapTimeDelta"] = df.groupby("Driver")["LapTime"].diff()

    threshold = df["LapTime"].mean() + 3 * df["LapTime"].std()
    df["IsPitStop"] = df["LapTime"] > threshold

    df["Stint"] = df.groupby("Driver")["IsPitStop"].cumsum() + 1
    df["DegradationRate"] = df["LapTimeDelta"] / df["TyreLife"]

    return df

# ===============================
# SUSTAINABILITY SCORE
# ===============================
def calculate_sustainability_score(df, driver):
    d = df[df["Driver"] == driver].copy()

    stint_lengths = d.groupby("Stint")["LapNumber"].count()
    avg_stint_length = stint_lengths.mean()

    valid_degradation = (
        d["DegradationRate"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    degr_rate_mean = valid_degradation.mean() if len(valid_degradation) else 0

    avg_lap_time = d["LapTime"].mean()
    pit_laps = d[d["IsPitStop"]]["LapTime"]
    pit_loss = (pit_laps - avg_lap_time).mean() if len(pit_laps) else 0

    score = (
        (avg_stint_length / abs(degr_rate_mean)) - pit_loss
        if degr_rate_mean != 0
        else 0
    )

    return {
        "AvgStintLength": avg_stint_length,
        "DegradationRate": degr_rate_mean,
        "SustainabilityScore": score
    }

# ===============================
# SIDEBAR – RACE & DRIVER SELECTION
# ===============================
st.sidebar.header("🏁 Race Selection")

selected_race_name = st.sidebar.selectbox("Select Race", race_names)
selected_race_file = race_files[race_names.index(selected_race_name)]

df_raw = load_data(selected_race_file)
df = prepare_features(df_raw)

drivers = sorted(df["Driver"].unique())

st.sidebar.header("👤 Driver Selection")
driver1 = st.sidebar.selectbox("Driver 1", drivers)
driver2 = st.sidebar.selectbox("Driver 2", drivers, index=1)

# ===============================
# DASHBOARD
# ===============================
st.subheader(f"Sustainability Performance — {selected_race_name}")

score1 = calculate_sustainability_score(df, driver1)
score2 = calculate_sustainability_score(df, driver2)

score_df = pd.DataFrame([
    {"Driver": driver1, **score1},
    {"Driver": driver2, **score2}
])

st.dataframe(score_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(score_df["Driver"], score_df["SustainabilityScore"])
ax.set_title("Sustainability Score Comparison")
ax.set_ylabel("Score")
ax.grid(axis="y", linestyle="--")

st.pyplot(fig)

# ===============================
# BUILD RACE SUMMARY FOR LLM
# ===============================
race_summary = f"""
Race: {selected_race_name}

{driver1} achieved an average stint length of {round(score1['AvgStintLength'], 2)} laps
with an average tyre degradation rate of {round(score1['DegradationRate'], 4)}.

{driver2} achieved an average stint length of {round(score2['AvgStintLength'], 2)} laps
with an average tyre degradation rate of {round(score2['DegradationRate'], 4)}.

Longer stints and lower degradation indicate better resource efficiency
and fewer operational interventions.
"""

# ===============================
# ESG CHATBOT (FIXED PROMPT)
# ===============================
st.subheader("💬 Ask the ESG Copilot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input(
    "Ask about ESG, sustainability, or what this race teaches companies"
)

if user_question:
    esg_context = retrieve_context(user_question)

    prompt = f"""
You are a corporate ESG sustainability assistant.
Explain concepts clearly and practically for business users.

Use the information below to answer the question.
Do not use numbering or bullet points unless necessary.
If the question is simple, give a simple explanation.

ESG Knowledge:
{esg_context}

Race Context:
{race_summary}

Question:
{user_question}

Answer:
"""

    response = llm(prompt)[0]["generated_text"].strip()

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("assistant", response))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "This project uses Formula 1 data as a sustainability case study. "
    "All ESG insights are for demonstration purposes only."
)
