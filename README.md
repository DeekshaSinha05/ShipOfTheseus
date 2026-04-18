# Ship of Theseus — Paraphrased Corpus Demo

An interactive Streamlit app for the *Ship of Theseus* computational linguistics research project, exploring how iterative LLM paraphrasing erodes authorial identity.

## Live Demo

[Launch on Streamlit Cloud](https://shipoftheseus.streamlit.app) *(update link after deployment)*

## Overview

The app investigates whether a text remains "the same" after repeated paraphrasing — and at what point authorship attribution breaks down. It includes five interactive tabs:

| Tab | Description |
|-----|-------------|
| **The Paradox** | Introduction and key findings |
| **Style Decay** | Multi-modal audit of syntactic and semantic drift across paraphrase iterations |
| **Style-Drift Explorer** | Live demo — explore drift metrics and t-SNE embeddings |
| **Point of No Return** | Authorship attribution decay across iterations (T0–T3) |
| **Paraphraser Fingerprints** | Forensic stylometry + live paraphraser predictor |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

> On first run, the app automatically downloads two large asset files (`rf_model.pkl`, `trajectory.csv`) from Hugging Face Hub.

## Project Structure

```
app.py                  # Streamlit app
requirements.txt        # Python dependencies
streamlit_assets/       # Precomputed data and models (small files)
```

Large files (`rf_model.pkl`, `trajectory.csv`) are hosted on [Hugging Face](https://huggingface.co/datasets/deekshaSinha/ship-of-theseus-assets-model-results) and downloaded automatically at startup.
