# Sustainability-copilot

# 🏎️🌱 ESG F1 Copilot  
**A Zero-Cost LLM-Powered Sustainability Assistant using Formula 1 Data**

ESG F1 Copilot is an interactive sustainability intelligence tool that combines a **data-driven dashboard** with a **lightweight LLM-based chatbot** to demonstrate how **operational performance data** can be translated into **corporate ESG insights**.

The project uses **Formula 1 race telemetry** as a **high-performance case study** to showcase how efficiency, resource usage, and strategic decisions impact sustainability outcomes — a concept directly transferable to manufacturing, energy, and industrial organizations.

---

## 🎯 Project Objectives

- Demonstrate how **operational data** can inform ESG reasoning
- Translate **performance metrics → sustainability insights**
- Build a **zero-cost, deployable LLM application**
- Avoid hallucinations by grounding AI responses in curated knowledge
- Showcase end-to-end product thinking (data → dashboard → AI insights)

---

## 🧠 Key Features

### 📊 Sustainability Dashboard (F1 Case Study)
- Race-wise data selection
- Driver-to-driver comparison
- Custom **Sustainability Performance Score** based on:
  - Average stint length
  - Tyre degradation rate
  - Pit stop time loss
- Visual comparison of sustainability efficiency

### 💬 ESG Copilot (LLM-Based Chatbot)
- Answers ESG and sustainability questions
- Uses **Formula 1 metrics as an analogy** for corporate operations
- Grounded using a curated ESG knowledge base (no hallucinated data)
- Context-aware: references the selected race and drivers when relevant

---

## 🏗️ System Architecture

User Question
↓
TF-IDF Retrieval (ESG Knowledge)
↓
Optional Race Summary Context
↓
Lightweight Open-Source LLM
↓
Business-Focused ESG Explanation


This follows a **Retrieval-Augmented Generation (RAG)** pattern, optimized for free infrastructure.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| UI | Streamlit |
| LLM | google/flan-t5-small |
| Retrieval | TF-IDF + Cosine Similarity |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Hosting | Streamlit Community Cloud |
| Cost | ₹0 / $0 |

---


---

## 📈 Sustainability Logic (Conceptual Mapping)

| Formula 1 Metric | ESG Interpretation |
|----------------|-------------------|
| Tyre degradation | Resource consumption efficiency |
| Stint length | Asset lifecycle optimization |
| Pit stops | Operational waste & interventions |
| Strategy stability | Governance & decision quality |
| Sustainability Score | Operational ESG KPI |

---

## 🧪 Example Questions to Try

-What is ESG and why does it matter?

-How does tyre degradation relate to sustainability?

-Compare the two drivers from a sustainability perspective

-What can manufacturing companies learn from this race?

-How does operational efficiency support ESG goals?

## ⚙️ Design Decisions & Trade-offs

-TF-IDF instead of neural embeddings
Chosen to ensure stability and deployability on free infrastructure.

-Small open-source LLM
Optimized for CPU inference with acceptable latency.

-Curated knowledge base
Prevents hallucinations and ensures explainable ESG reasoning.

-Formula 1 as an analogy
High-performance environments mirror real-world industrial constraints.

## ⚠️ Disclaimer

This project uses Formula 1 data as a conceptual sustainability case study.
All ESG insights are illustrative and not intended as financial, legal, or investment advice.

## 👤 Author

Built as a portfolio project to explore AI, sustainability, and data-driven decision-making using open-source tools and zero-cost infrastructure.
