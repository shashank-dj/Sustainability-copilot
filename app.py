import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline

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
# EMBEDDINGS + FAISS
# ===============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

def retrieve_context(query, k=2):
    q_emb = embedder.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return "\n".join([documents[i] for i in idx[0]])

# ===============================
# LOAD LLM
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
    df["SectorTotal"] = df["Sector1Time"] + df["Sector2Time"] + df["Sector3Time"]
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
    d = df[df["Driver"] == driver]

    stint_lengths = d.groupby("Stint")["LapNumber"].count()
    avg_stint = stint_lengths.mean()

    degr = d["DegradationRate"].replace([np.inf, -np.inf], np.nan).dropna()
    degr_mean = degr.mean() if len(degr) else 0

    pit_laps = d[d["IsPitStop"]]["LapTime"]
    pit_loss = (pit_laps - d["LapTime"].mean()).mean() if len(pit_laps) else 0

    score = (avg_stint / abs(degr_mean)) - pit_loss if degr_mean != 0 else 0

    return avg_stint, degr_mean, score

# ===============================
# UI
# ===============================
st.set_page_config(layout="wide")
st.title("🏎️🌱 ESG F1 Copilot")

# Sidebar
st.sidebar.header("Race Selection")
selected_race_name = st.sidebar.selectbox("Select Race", race_names)
selected_file = race_files[race_names.index(selected_race_name)]

df = prepare_features(load_data(selected_file))

drivers = sorted(df["Driver"].unique())
d1 = st.sidebar.selectbox("Driver 1", drivers)
d2 = st.sidebar.selectbox("Driver 2", drivers, index=1)

# ===============================
# DASHBOARD (UNCHANGED CORE)
# ===============================
st.subheader("Sustainability Score Comparison")

s1 = calculate_sustainability_score(df, d1)
s2 = calculate_sustainability_score(df, d2)

score_df = pd.DataFrame({
    "Driver": [d1, d2],
    "Avg Stint Length": [s1[0], s2[0]],
    "Degradation Rate": [s1[1], s2[1]],
    "Sustainability Score": [s1[2], s2[2]]
})

st.dataframe(score_df)

# ===============================
# BUILD RACE SUMMARY FOR LLM
# ===============================
race_summary = f"""
In the {selected_race_name},
{d1} had an average stint length of {round(s1[0],2)} laps
with an average degradation rate of {round(s1[1],4)}.

{d2} had an average stint length of {round(s2[0],2)} laps
with an average degradation rate of {round(s2[1],4)}.

Higher stint length and lower degradation indicate better
resource efficiency and reduced operational waste.
"""

# ===============================
# CHATBOT
# ===============================
st.subheader("💬 Ask the ESG Copilot")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.chat_input("Ask about ESG, sustainability, or performance insights")

if question:
    esg_context = retrieve_context(question)

    prompt = f"""
You are a corporate ESG sustainability assistant.
Use Formula 1 data as an analogy for business operations.

ESG Knowledge:
{esg_context}

Race Context:
{race_summary}

Question:
{question}

Answer with:
- Explanation
- Business relevance
- ESG takeaway
"""

    answer = llm(prompt)[0]["generated_text"]
    st.session_state.chat.append((question, answer))

for q, a in st.session_state.chat:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# ===============================
# FOOTER
# ===============================
st.caption("Demonstration project using F1 data as a sustainability case study.")
