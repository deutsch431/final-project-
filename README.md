# 🛒 Smart Product Intelligence — Electronics

> **Khazar University | Hands-on Deep Learning | Capstone Project**
> An end-to-end deep learning system for Amazon Electronics products — tabular ML, computer vision, NLP, LLMs, and generative AI in one unified application.

---

## 📌 Project Overview

This project builds a **Smart Product Assistant** for the Amazon Electronics category using the [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset. Every major deep learning technique is applied to the same dataset across 8 milestones, culminating in a single Gradio demo app.

**Dataset:** `raw_review_Electronics` + `raw_meta_Electronics`
**Scope:** ~15,000 unique products · ~50,000 reviews · ~7,000 cached images

---

## 🗂️ Repository Structure

```
smart-product-intelligence/
├── README.md
├── requirements.txt
├── .env.example
│
├── data/                        # ← gitignored (cached locally)
│   ├── reviews.parquet
│   ├── metadata.parquet
│   └── images/                  # downloaded product images
│
├── notebooks/
│   ├── 00_eda.ipynb             # M0 — Data Preparation & EDA
│   ├── 01_tabular_mlp.ipynb     # M1 — Tabular MLP
│   ├── 02_vision_cnn.ipynb      # M2 — Computer Vision (CNN + Transfer Learning)
│   ├── 03_text_embeddings.ipynb # M3 — Text Vectorization & Embeddings
│   ├── 04_transformers.ipynb    # M4 — BERT / DistilBERT Fine-tuning
│   ├── 05_llm_rag.ipynb         # M5 — LLMs, RAG & Fine-tuning
│   └── 06_diffusion.ipynb       # M6 — Stable Diffusion Image Generation
│
├── src/
│   ├── data.py                  # Data loading, cleaning, splitting utils
│   ├── models.py                # Shared model definitions (MLP, CNN, etc.)
│   └── utils.py                 # Metrics, plotting, embedding helpers
│
├── app/
│   └── app.py                   # M7 — Gradio demo (Smart Product Assistant)
│
└── report/
    └── final_report.pdf
```

---

## 🧩 Milestones

| # | Milestone | Technique | Weight |
|---|-----------|-----------|--------|
| 0 | Data Preparation & EDA | Pandas, Hugging Face datasets | — |
| 1 | Tabular MLP | Keras MLP vs. Linear Regression | 10% |
| 2 | Computer Vision | CNN from scratch vs. MobileNetV2/EfficientNet | 15% |
| 3 | Text & Embeddings | TF-IDF → Learned Embeddings + Semantic Search | 15% |
| 4 | Transformers | DistilBERT fine-tuning (review classification) | 15% |
| 5 | LLMs & RAG | Review summarizer + Grounded Q&A (RAG) | 20% |
| 6 | Diffusion | Stable Diffusion product image generation | 10% |
| 7 | Integration & Demo | Gradio app combining all milestones | 15% |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/<your-username>/smart-product-intelligence.git
cd smart-product-intelligence
pip install -r requirements.txt
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys (HuggingFace token, optional OpenAI key)
```

### 3. Download & cache data (run once)
```bash
# Inside 00_eda.ipynb — or run the script directly:
python src/data.py --download --category Electronics --max-products 15000
```

### 4. Run notebooks in order
Open notebooks in `notebooks/` sequentially: `00_eda.ipynb` → `01_tabular_mlp.ipynb` → ...

### 5. Launch the demo app
```bash
cd app
python app.py
# Opens Gradio UI at http://localhost:7860
```

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Core DL | Keras / TensorFlow 2.x | MLP, CNN, Transformer training |
| Dataset | `datasets` (Hugging Face) | Loading Amazon Reviews 2023 |
| Pretrained models | `transformers` (Hugging Face) | BERT, DistilBERT, LLMs |
| Generative | `diffusers` | Stable Diffusion image generation |
| Embeddings | `sentence-transformers` | Semantic product search |
| Compute | Google Colab / Kaggle (free GPU) | All milestones fit on free tier |
| Demo UI | Gradio | Interactive web app |
| Experiment tracking | Matplotlib / seaborn | Learning curves, confusion matrices |

---

## ⚠️ Key Rules (avoid common pitfalls)

- **Split by product**, not by review — prevents data leakage across splits
- **Cache images in M0** — never fetch live during training (links go stale)
- **Always build a baseline first** — Linear/Logistic Regression before any deep model
- **Use macro-F1**, not accuracy — ratings are heavily imbalanced (mostly 5★)
- **M7 assembles earlier outputs** — do not introduce new models in the final milestone

---

## 📊 Data Overview

```
Source : McAuley-Lab/Amazon-Reviews-2023 → Electronics
Reviews: ~50,000 (text, rating, helpful votes, timestamps)
Products: ~15,000 (price, categories, description, image URLs)
Images : ~7,000 downloaded locally (JPG/PNG)
Splits : train / val / test — split by parent_asin (product ID)
```

---

## 📋 Reproducibility

A clean clone + `pip install -r requirements.txt` should reproduce all results.
- Random seeds are fixed in every notebook (`SEED = 42`)
- Frozen splits are saved as `data/splits.json` after M0
- Model checkpoints are saved in `data/checkpoints/` (gitignored, regeneratable)

---

## 👤 Author

**[Your Name]** — Khazar University, Hands-on Deep Learning, 2024–2025
