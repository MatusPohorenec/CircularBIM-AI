# 🏗️ BIM–LCA–Circularity AI

> **AI-Based Predictive Integrated Model for BIM–LCA–Circularity Assessment in Construction Projects**

An interactive Streamlit dashboard that predicts sustainability outcomes (recycling rate, waste reduction, CO₂ emission reduction) from BIM adoption profiles using machine learning, and compares adoption strategies via TOPSIS multi-criteria decision making.

---

## 📸 Screenshots

| Landing Page | Explorer | Predictor |
|:---:|:---:|:---:|
| *screenshot placeholder* | *screenshot placeholder* | *screenshot placeholder* |

| Model Insights | Scenario MCDM | About |
|:---:|:---:|:---:|
| *screenshot placeholder* | *screenshot placeholder* | *screenshot placeholder* |

---

## ✨ Features

- **Data Explorer** — Interactive heatmaps, radar charts, correlation matrices, and distribution plots from survey data across Slovakia, Croatia, and Slovenia (199 respondents)
- **Sustainability Predictor** — Configure a BIM adoption profile and get real-time predictions with gauges, probability distributions, and natural-language interpretation
- **Model Insights** — Full transparency layer: model comparison table, SHAP feature importance, PLS-SEM path coefficients, construct reliability
- **Scenario MCDM** — Compare 6 predefined + 1 custom BIM adoption scenario using TOPSIS with adjustable criterion weights and sensitivity analysis
- **About** — Methodology, dataset summary, limitations, and citation

---

## 🧪 Methodology

Six-phase research pipeline:

| Phase | Description |
|---|---|
| **1** | Data Collection & Exploratory Data Analysis |
| **2** | Construct Validation (Cronbach's α, EFA, CFA, AVE, HTMT) |
| **3** | Inferential Modelling (Ordinal Logistic Regression, PLS-SEM) |
| **4** | Predictive AI (Random Forest, XGBoost, LightGBM, SVM, mord) |
| **5** | MCDM Scenario Analysis (TOPSIS with sensitivity) |
| **6** | Validation & Self-Challenge |

---

## 🚀 Local Development

### Prerequisites

- [Python 3.12+](https://www.python.org/)
- [UV](https://docs.astral.sh/uv/) (recommended) or pip

### Setup with UV

```bash
# Clone the repository
git clone <repo-url>
cd sustainability-24.02.25

# Install dependencies
uv sync

# Run the full analysis pipeline (optional — pre-computed assets are included)
uv run python main.py

# Start the Streamlit app
uv run streamlit run app.py
```

### Setup with pip

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
streamlit run app.py
```

The app will be available at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repository
4. Set the main file path to `app.py`
5. Click **Deploy**

> **Note:** The app runs in inference-only mode — all ML models and aggregate data are pre-computed and stored in `assets/` and `models/`. No raw survey data is required at runtime.

---

## 📁 Project Structure

```
├── app.py                    # Landing page
├── pages/
│   ├── 1_🔍_Explorer.py      # Data explorer dashboard
│   ├── 2_🤖_Predictor.py     # Interactive prediction tool
│   ├── 3_📊_Model_Insights.py # Model transparency layer
│   ├── 4_🏗️_Scenario_MCDM.py # TOPSIS scenario comparison
│   └── 5_📄_About.py         # Methodology & citation
├── src/
│   ├── charts.py             # Plotly chart builders
│   ├── model.py              # Model loading & inference
│   ├── st_config.py          # Streamlit constants & theming
│   ├── config.py             # Pipeline configuration
│   ├── eda.py                # Exploratory data analysis
│   ├── construct_validation.py
│   ├── inferential_modelling.py
│   ├── predictive_modelling.py
│   ├── mcdm.py               # TOPSIS implementation
│   ├── validation.py         # Post-hoc validation
│   └── data_loading.py       # Data ingestion & cleaning
├── assets/                   # Pre-computed JSON aggregates
├── models/                   # Serialised .joblib models
├── scripts/
│   └── export_models.py      # Model export utility
├── main.py                   # Full pipeline runner
├── pyproject.toml            # Project metadata & dependencies
├── requirements.txt          # Deployment dependencies
└── .streamlit/config.toml    # Streamlit theme configuration
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Streamlit 1.54 |
| Charting | Plotly (dark theme) |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Interpretability | SHAP (TreeExplainer) |
| MCDM | TOPSIS (custom implementation) |
| SEM | semopy |
| Serialisation | joblib |

---

## ⚠️ Limitations

- Sustainability targets are **perceived impacts** (survey-based), not measured LCA/GWP values
- The dataset contains only **9 unique respondent profiles** across 199 rows, limiting effective sample diversity
- Perfect CV scores (F1 = 1.0) indicate potential overfitting to the limited profile set
- Country subgroups are unbalanced: Croatia (99), Slovakia (60), Slovenia (40)
- The Circularity sub-construct (2 items) has low internal consistency (α = 0.37)

---

## 📝 Citation

```bibtex
@article{bim_lca_circularity_2025,
  title   = {AI-Based Predictive Integrated Model for BIM--LCA--Circularity
             Assessment in Construction Projects},
  author  = {[Author Names]},
  journal = {[Target Journal]},
  year    = {2025},
  note    = {Manuscript in preparation}
}
```

---

## 📄 License

This project is part of academic research. Please contact the authors for licensing information.
