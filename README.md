# Ship of Theseus — Paraphrased Corpus Demo

An interactive Streamlit app for the *Ship of Theseus* computational linguistics research project, exploring how iterative LLM paraphrasing erodes authorial identity.

## Live Demo

[Launch on Streamlit Cloud](https://shipoftheseus.streamlit.app)

## Overview

The app investigates whether a text remains "the same" after repeated paraphrasing — and at what point authorship attribution breaks down. It includes four interactive tabs:

| Tab | Description |
|-----|-------------|
| **Overview** | Key findings and summary statistics |
| **Datasets** | Corpus breakdown and data schema |
| **Style Decay** | Multi-modal audit of syntactic and lexical drift across paraphrase iterations (T0–T3) |
| **Drift Explorer** | Live side-by-side T0 vs T3 comparison with SBERT cosine score and t-SNE embeddings |

## Project Structure

```
ShipOfTheseus/
├── README.md
├── requirements.txt
├── notebooks/
│   └── NLP_FinalSubmission_Team5.ipynb
├── paper/
│   └── The_Ship_of_Theseus_...pdf
├── streamlit_app/
│   ├── app.py
│   └── streamlit_assets/       # Precomputed CSVs, JSONs, models
├── data/
│   └── processed/              # Paraphrased corpus CSVs (gitignored)
├── figures/                    # Output charts
└── docs/                       # Project documentation
```

## Setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

On first run, the app automatically downloads large asset files (`trajectory.csv`, `rf_model.pkl`) from Hugging Face Hub.

Large files are hosted on [Hugging Face](https://huggingface.co/datasets/deekshaSinha/ship-of-theseus-assets-model-results).

## Development Note

All experiments and model training were developed on **Google Colab Pro**, which provides a single persistent notebook environment with GPU access. As a result, the full pipeline — data processing, feature extraction, model training, and evaluation — is contained in a single notebook (`notebooks/NLP_FinalSubmission_Team5.ipynb`) rather than split across multiple files.
