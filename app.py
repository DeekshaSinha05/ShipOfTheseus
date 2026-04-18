"""
Ship of Theseus — Paraphrased Corpus Demo
Streamlit app for the computational linguistics research defense.
Run: streamlit run app.py   (from inside ship_of_theseus/)
"""

import json
import pickle
import re
import string
from collections import Counter

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Ship of Theseus",
    page_icon="⚓",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
ITER_COLORS = {"T0": "#2196F3", "T1": "#4CAF50", "T2": "#FF9800", "T3": "#F44336"}
ITER_ORDER = ["T0", "T1", "T2", "T3"]
GROUP_COLORS = {"Human": "#2196F3", "LLM": "#F44336"}
PARAPHRASER_PALETTE = px.colors.qualitative.Set2   # 8-colour safe for 6 paraphrasers

ASSETS = "streamlit_assets"
HF_REPO = "deekshaSinha/ship-of-theseus-assets-model-results"
HF_LARGE_FILES = ["rf_model.pkl", "trajectory.csv"]

def _ensure_large_assets():
    os.makedirs(ASSETS, exist_ok=True)
    for fname in HF_LARGE_FILES:
        if not os.path.exists(f"{ASSETS}/{fname}"):
            with st.spinner(f"Downloading {fname} from Hugging Face…"):
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=fname,
                    repo_type="dataset",
                    local_dir=ASSETS,
                )

_ensure_large_assets()

# ── Data loaders (all cached) ──────────────────────────────────────────────────

@st.cache_data
def load_summary_stats() -> dict:
    with open(f"{ASSETS}/summary_stats.json") as f:
        return json.load(f)

@st.cache_data
def load_clf_results() -> dict:
    with open(f"{ASSETS}/clf_results.json") as f:
        return json.load(f)

@st.cache_data
def load_ds_fingerprint_f1() -> dict:
    with open(f"{ASSETS}/ds_fingerprint_f1.json") as f:
        return json.load(f)

@st.cache_data
def load_attribution_f1() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/attribution_f1.csv")

@st.cache_data
def load_pos_cosine_drift() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/pos_cosine_drift.csv")

@st.cache_data
def load_sbert_cosine_drift() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/sbert_cosine_drift.csv")

@st.cache_data
def load_tsne_coords() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/tsne_coords.csv")

@st.cache_data
def load_trajectory() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/trajectory.csv")

@st.cache_data
def load_linguistic_delta() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/linguistic_delta.csv")

@st.cache_data
def load_fingerprint_features() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/fingerprint_features.csv")

@st.cache_resource
def load_rf_model():
    with open(f"{ASSETS}/rf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(f"{ASSETS}/scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoder():
    with open(f"{ASSETS}/label_encoder.pkl", "rb") as f:
        return pickle.load(f)


# ── Feature extraction (no spaCy) ─────────────────────────────────────────────

def extract_text_features(text: str) -> dict:
    """Compute lexical features from raw text. POS/dep features are set to 0."""
    text = str(text).strip()
    if not text:
        return dict(ttr=0.0, avg_sent_len=0.0, avg_word_len=0.0,
                    hapax_rate=0.0, punct_rate=0.0, n_sents=0.0, n_words=0.0,
                    noun_rate=0.0, verb_rate=0.0, dep_depth=0.0)

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    n_words = max(len(words), 1)
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sents = max(len(sents), 1)

    word_counts = Counter(words)
    ttr = len(set(words)) / n_words
    hapax_rate = sum(1 for c in word_counts.values() if c == 1) / n_words
    avg_sent_len = n_words / n_sents
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    punct_rate = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)

    return dict(
        ttr=ttr, avg_sent_len=avg_sent_len, avg_word_len=avg_word_len,
        hapax_rate=hapax_rate, punct_rate=punct_rate,
        n_sents=float(n_sents), n_words=float(n_words),
        noun_rate=0.0, verb_rate=0.0, dep_depth=0.0,
    )


def build_feature_vector(feats: dict, n_expected: int) -> np.ndarray:
    """Build numpy feature vector respecting the number of features the scaler expects."""
    order_10 = ["ttr", "avg_sent_len", "avg_word_len", "hapax_rate", "punct_rate",
                 "n_sents", "n_words", "noun_rate", "verb_rate", "dep_depth"]
    order_8 = ["ttr", "avg_sent_len", "avg_word_len", "hapax_rate", "punct_rate",
                "noun_rate", "verb_rate", "dep_depth"]
    order_6 = ["ttr", "avg_sent_len", "hapax_rate", "noun_rate", "verb_rate", "dep_depth"]

    if n_expected == 10:
        order = order_10
    elif n_expected == 8:
        order = order_8
    elif n_expected == 6:
        order = order_6
    else:
        order = order_10[:n_expected]

    return np.array([[feats.get(k, 0.0) for k in order]])


# ── Helper: drift stats ────────────────────────────────────────────────────────

def compute_drift_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per iteration, prepending a T0 baseline of 1.0."""
    rows = [{"Iteration": "T0", "Mean": 1.0, "Std": 0.0}]
    for col in ["T1", "T2", "T3"]:
        if col in df.columns:
            vals = df[col].dropna()
            rows.append({"Iteration": col, "Mean": vals.mean(), "Std": vals.std()})
    return pd.DataFrame(rows)


def drift_chart(stats: pd.DataFrame, title: str, color: str, y_min: float) -> go.Figure:
    """Line + shaded error-band figure for cosine drift."""
    fig = go.Figure()
    # Lower bound (rendered first so tonexty fills up to upper)
    fig.add_trace(go.Scatter(
        x=stats["Iteration"], y=stats["Mean"] - stats["Std"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # Upper bound filled to previous
    fig.add_trace(go.Scatter(
        x=stats["Iteration"], y=stats["Mean"] + stats["Std"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.15,)}",
        name="± 1 SD", hoverinfo="skip",
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=stats["Iteration"], y=stats["Mean"],
        mode="lines+markers", name="Mean",
        line=dict(color=color, width=2.5), marker=dict(size=9),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Cosine Similarity",
        yaxis=dict(range=[y_min, 1.03]),
        template="plotly_white",
        legend=dict(orientation="h", y=1.08),
        height=370,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    summary = load_summary_stats()

    # ── Global CSS tweak ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .kpi-card {padding:1rem 1.2rem; border-radius:10px;
                   background:#f7f9fc; border:1px solid #e0e4ea; text-align:center;}
        .kpi-label {font-size:0.82rem; color:#666; text-transform:uppercase; letter-spacing:.05em;}
        .kpi-value {font-size:2rem; font-weight:700; color:#1a1a2e;}
        .insight-box {padding:0.9rem 1.2rem; border-radius:8px;
                      background:#fffde7; border-left:4px solid #f9a825;}
        .forensic-box {padding:0.9rem 1.2rem; border-radius:8px;
                       background:#fce4ec; border-left:4px solid #c62828;
                       font-style:italic;}
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⚓ Overview",
        "📚 Datasets",
        "📉 Style Decay",
        "🔍 Style-Drift Explorer",
        "🚨 Point of No Return",
        "🕵️ Paraphraser Fingerprints",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.title("The Paradox of Authorial Identity")
        st.caption("Ship of Theseus Paraphrased Corpus  ·  NLP Research Defense · April 20, 2026")
        st.divider()

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        n_docs = summary.get("n_documents", 0)
        sbert_t3 = summary.get("mean_sbert_t3", None)
        f1_t0 = summary.get("attr_f1_t0", None)
        f1_t3 = summary.get("attr_f1_t3", None)

        with c1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Total Documents</div>'
                f'<div class="kpi-value">{n_docs:,}</div></div>', unsafe_allow_html=True)
        with c2:
            val = f"{sbert_t3:.4f}" if sbert_t3 is not None else "N/A"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Mean SBERT Cosine @ T3</div>'
                f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
        with c3:
            val = f"{f1_t0:.4f}" if f1_t0 is not None else "N/A"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Authorship F1 @ T0</div>'
                f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
        with c4:
            val = f"{f1_t3:.4f}" if f1_t3 is not None else "N/A"
            delta = ""
            if f1_t0 is not None and f1_t3 is not None:
                d = f1_t3 - f1_t0
                arrow = "▲" if d >= 0 else "▼"
                delta = f'<div style="font-size:.9rem;color:{"green" if d>=0 else "red"};">{arrow} {abs(d):.4f}</div>'
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Authorship F1 @ T3</div>'
                f'<div class="kpi-value">{val}</div>{delta}</div>', unsafe_allow_html=True)

        st.divider()

        datasets = summary.get("datasets", [])
        paraphrasers = summary.get("paraphrasers", [])
        sbert_val = sbert_t3 if sbert_t3 is not None else 0.87

        st.markdown("### Research Summary")
        st.markdown(f"""
The **Ship of Theseus** thought experiment asks: if every plank of a ship is replaced one by one,
is it still the same ship? This research applies that paradox to **authorial identity in text**.
Across **{len(datasets)} datasets** (*{', '.join(datasets)}*) and **{len(paraphrasers)} AI paraphrasers**
(*{', '.join(paraphrasers)}*), documents are iteratively paraphrased up to **three times**
(T0 → T1 → T2 → T3), tracking whether authorial style — measured via syntactic POS structure,
SBERT semantic embeddings, and lexical richness metrics (TTR, hapax rate) — survives progressive
AI rewriting. The central paradox: semantics remain surprisingly resilient
(mean SBERT cosine ≈ **{sbert_val:.3f}** at T3), yet paraphrasers leave forensically detectable
linguistic fingerprints, raising the fundamental question — *after three rounds of AI rewriting,
whose identity does the text ultimately bear: the original author's, or the machine's?*
        """)

        col_ds, col_par = st.columns(2)
        with col_ds:
            st.markdown(f"**Datasets ({len(datasets)}):** " + " · ".join(f"`{d}`" for d in datasets))
        with col_par:
            st.markdown(f"**Paraphrasers ({len(paraphrasers)}):** " + " · ".join(f"`{p}`" for p in paraphrasers))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DATASETS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("Dataset Explorer")
        st.markdown("*Understand the seven source corpora used in this study.*")
        st.divider()

        DATASET_INFO = {
            "cmv": {
                "full_name": "Change My View (CMV)",
                "source": "Reddit r/changemyview",
                "domain": "Argumentative / Persuasion",
                "src_group": "Human",
                "description": (
                    "Long-form Reddit posts where users present an opinion and invite others to "
                    "challenge it. Rich in structured argumentation, hedging language, and "
                    "complex sentence constructions — making it a demanding stylometric target."
                ),
            },
            "eli5": {
                "full_name": "Explain Like I'm 5 (ELI5)",
                "source": "Reddit r/explainlikeimfive",
                "domain": "Explanatory / Informal",
                "src_group": "Human",
                "description": (
                    "Crowd-sourced simplified explanations of complex topics. Characterised by "
                    "short sentences, high accessibility, use of analogies, and informal register. "
                    "Contrasts sharply with academic corpora in lexical richness."
                ),
            },
            "sci_gen": {
                "full_name": "Scientific Paper Abstracts (SciGen)",
                "source": "arXiv / Academic Papers",
                "domain": "Scientific / Technical",
                "src_group": "LLM",
                "description": (
                    "Abstracts from scientific publications spanning multiple disciplines. "
                    "Dense noun phrases, passive constructions, and high type-token ratios "
                    "are hallmarks. LLM-generated subset tests whether machine-written science "
                    "is stylistically distinguishable from human-written science."
                ),
            },
            "tldr": {
                "full_name": "TL;DR Summaries",
                "source": "Reddit (user-written summaries)",
                "domain": "Summary / Informal",
                "src_group": "Human",
                "description": (
                    "User-generated 'too long; didn't read' summaries of Reddit posts. "
                    "Extremely compressed, colloquial, and often telegraphic — providing a "
                    "low word-count edge case for stylometric analysis."
                ),
            },
            "wp": {
                "full_name": "Writing Prompts (WP)",
                "source": "Reddit r/WritingPrompts",
                "domain": "Creative Fiction",
                "src_group": "Human",
                "description": (
                    "Creative short stories written in response to community prompts. "
                    "High narrative variety, broad vocabulary, and expressive stylistic "
                    "choices — one of the richest corpora for authorship attribution."
                ),
            },
            "xsum": {
                "full_name": "Extreme Summarisation (XSum)",
                "source": "BBC News articles",
                "domain": "News / Journalistic",
                "src_group": "Human",
                "description": (
                    "Single-sentence summaries of BBC news articles. Formal journalistic "
                    "register, consistent style conventions, and factual tone. Very short "
                    "texts challenge paraphrasers to preserve meaning in minimal tokens."
                ),
            },
            "yelp": {
                "full_name": "Yelp Reviews",
                "source": "Yelp Open Dataset",
                "domain": "Review / Opinion",
                "src_group": "Human",
                "description": (
                    "Consumer reviews spanning restaurants and businesses. Highly subjective, "
                    "emotionally varied (positive / negative), and colloquial — providing "
                    "sentiment-driven stylistic signals that are sensitive to paraphrasing."
                ),
            },
        }

        GROUP_BADGE = {
            "Human": '<span style="background:#e3f2fd;color:#1565c0;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;">Human</span>',
            "LLM":   '<span style="background:#fce4ec;color:#b71c1c;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;">LLM</span>',
        }

        fp_df_ds = load_fingerprint_features()

        # ── Per-dataset stats ────────────────────────────────────────────────
        ds_stats = (
            fp_df_ds.groupby("dataset")
            .agg(
                n_samples=("ttr", "count"),
                mean_ttr=("ttr", "mean"),
                mean_words=("n_words", "mean"),
                mean_sent_len=("avg_sent_len", "mean"),
            )
            .reset_index()
        )

        # ── Dataset cards ────────────────────────────────────────────────────
        ds_keys = list(DATASET_INFO.keys())
        for i in range(0, len(ds_keys), 2):
            cols = st.columns(2)
            for j, ds in enumerate(ds_keys[i:i+2]):
                info = DATASET_INFO[ds]
                row = ds_stats[ds_stats["dataset"] == ds]
                n = int(row["n_samples"].values[0]) if not row.empty else 0
                ttr = f"{row['mean_ttr'].values[0]:.3f}" if not row.empty else "—"
                words = f"{row['mean_words'].values[0]:.0f}" if not row.empty else "—"
                badge = GROUP_BADGE.get(info["src_group"], "")
                with cols[j]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #e0e4ea;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.5rem;background:#fafbfc;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:1.05rem;font-weight:700;">{info['full_name']}</span>
                            {badge}
                        </div>
                        <div style="font-size:0.78rem;color:#888;margin:2px 0 6px 0;">{info['source']} &nbsp;·&nbsp; {info['domain']}</div>
                        <div style="font-size:0.88rem;color:#444;margin-bottom:10px;">{info['description']}</div>
                        <div style="display:flex;gap:1.5rem;">
                            <div><span style="font-size:0.75rem;color:#888;">SAMPLES</span><br><strong>{n:,}</strong></div>
                            <div><span style="font-size:0.75rem;color:#888;">MEAN TTR</span><br><strong>{ttr}</strong></div>
                            <div><span style="font-size:0.75rem;color:#888;">MEAN WORDS</span><br><strong>{words}</strong></div>
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ── Charts row ───────────────────────────────────────────────────────
        cc1, cc2 = st.columns(2)

        with cc1:
            fig_cnt = px.bar(
                ds_stats.sort_values("n_samples", ascending=False),
                x="dataset", y="n_samples",
                title="Sample Count per Dataset",
                labels={"dataset": "Dataset", "n_samples": "Samples"},
                color="n_samples", color_continuous_scale="Blues",
                text="n_samples",
            )
            fig_cnt.update_traces(textposition="outside")
            fig_cnt.update_layout(
                template="plotly_white", height=370,
                coloraxis_showscale=False, showlegend=False,
            )
            st.plotly_chart(fig_cnt, use_container_width=True)

        with cc2:
            fig_ttr_ds = px.box(
                fp_df_ds, x="dataset", y="ttr",
                title="Type-Token Ratio Distribution per Dataset",
                labels={"dataset": "Dataset", "ttr": "TTR"},
                color="dataset",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_ttr_ds.update_layout(
                template="plotly_white", height=370, showlegend=False,
            )
            st.plotly_chart(fig_ttr_ds, use_container_width=True)

        cc3, cc4 = st.columns(2)

        with cc3:
            fig_words_ds = px.box(
                fp_df_ds, x="dataset", y="n_words",
                title="Word Count Distribution per Dataset",
                labels={"dataset": "Dataset", "n_words": "Word Count"},
                color="dataset",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_words_ds.update_layout(
                template="plotly_white", height=370, showlegend=False,
            )
            st.plotly_chart(fig_words_ds, use_container_width=True)

        with cc4:
            para_dist = fp_df_ds.groupby(["dataset", "label"]).size().reset_index(name="count")
            fig_para = px.bar(
                para_dist, x="dataset", y="count", color="label",
                barmode="stack",
                title="Paraphraser Sample Distribution per Dataset",
                labels={"dataset": "Dataset", "count": "Samples", "label": "Paraphraser"},
                color_discrete_sequence=PARAPHRASER_PALETTE,
            )
            fig_para.update_layout(
                template="plotly_white", height=370,
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_para, use_container_width=True)

    # TAB 3 — STYLE DECAY
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:  # was tab2
        st.header("Style Decay — Multi-Modal Audit")
        st.markdown("*Structural skeleton vs. lexical skin: does syntax outlast vocabulary?*")
        st.divider()

        pos_df = load_pos_cosine_drift()
        sbert_df = load_sbert_cosine_drift()

        pos_stats = compute_drift_stats(pos_df)
        sbert_stats = compute_drift_stats(sbert_df)

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                drift_chart(pos_stats, "Syntactic Skeleton Stability", "#2196F3", y_min=0.85),
                use_container_width=True,
            )
        with col_r:
            st.plotly_chart(
                drift_chart(sbert_stats, "Semantic Identity Drift", "#F44336", y_min=0.55),
                use_container_width=True,
            )

        st.divider()

        ling_df = load_linguistic_delta()

        # Ensure iteration order is categorical
        if "iteration" in ling_df.columns:
            ling_df["iteration"] = pd.Categorical(
                ling_df["iteration"], categories=ITER_ORDER, ordered=True
            )
            ling_df = ling_df.sort_values(["paraphraser", "iteration"])

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            fig_rttr = px.line(
                ling_df, x="iteration", y="RTTR", color="paraphraser",
                markers=True,
                title="Root Type-Token Ratio per Paraphraser (T0→T3)",
                labels={"RTTR": "RTTR", "iteration": "Iteration", "paraphraser": "Paraphraser"},
                color_discrete_sequence=PARAPHRASER_PALETTE,
            )
            fig_rttr.update_layout(
                template="plotly_white", height=360,
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_rttr, use_container_width=True)

        with col_r2:
            fig_nw = px.line(
                ling_df, x="iteration", y="N Words", color="paraphraser",
                markers=True,
                title="Mean Word Count per Paraphraser (T0→T3)",
                labels={"N Words": "Word Count", "iteration": "Iteration", "paraphraser": "Paraphraser"},
                color_discrete_sequence=PARAPHRASER_PALETTE,
            )
            fig_nw.update_layout(
                template="plotly_white", height=360,
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_nw, use_container_width=True)

        st.markdown(
            '<div class="insight-box">'
            '<strong>Key Insight — Did the syntactic skeleton remain more stable than vocabulary?</strong><br>'
            'POS cosine similarity (structural) drops far less steeply than SBERT cosine (semantic) '
            'across three paraphrase iterations. AI paraphrasers preserve grammatical structure even as they '
            'substitute vocabulary — the <em>skeleton</em> survives the Ship of Theseus transformation, '
            'but the <em>lexical skin</em> does not.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — STYLE-DRIFT EXPLORER (LIVE DEMO)
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("Style-Drift Explorer — Live Demo")
        st.markdown("*Browse how original texts (T0) transform through three rounds of AI paraphrasing (T3).*")

        trajectory_df = load_trajectory()
        tsne_df = load_tsne_coords()

        # ── Filters ───────────────────────────────────────────────────────────
        with st.container():
            st.markdown("##### Filter Controls")
            fc1, fc2, fc3 = st.columns([2, 1, 2])

            with fc1:
                all_ds = sorted(trajectory_df["dataset"].dropna().unique().tolist())
                sel_ds = st.multiselect(
                    "Dataset", options=all_ds, default=all_ds, key="t3_ds"
                )
            with fc2:
                all_grp = sorted(trajectory_df["src_group"].dropna().unique().tolist())
                grp_options = ["Both"] + all_grp
                sel_grp = st.selectbox("Source Group", options=grp_options, index=0, key="t3_grp")
            with fc3:
                sbert_col = "sbert_cos_t3"
                valid_sbert = trajectory_df[sbert_col].dropna() if sbert_col in trajectory_df.columns else pd.Series([0.0, 1.0])
                s_min = float(valid_sbert.min()) if len(valid_sbert) else 0.0
                s_max = float(valid_sbert.max()) if len(valid_sbert) else 1.0
                sbert_range = st.slider(
                    "SBERT Cosine @ T3",
                    min_value=round(s_min, 2), max_value=round(s_max, 2),
                    value=(round(s_min, 2), round(s_max, 2)),
                    step=0.01, key="t3_sbert",
                )

        # Apply filters
        filt = trajectory_df.copy()
        if sel_ds:
            filt = filt[filt["dataset"].isin(sel_ds)]
        if sel_grp != "Both" and sel_grp in filt["src_group"].values:
            filt = filt[filt["src_group"] == sel_grp]
        if sbert_col in filt.columns:
            filt = filt[filt[sbert_col].between(sbert_range[0], sbert_range[1], inclusive="both")]

        st.divider()

        # ── Text comparison ───────────────────────────────────────────────────
        st.subheader("T0 vs T3 Side-by-Side")

        if filt.empty:
            st.warning(
                "No rows match the current filters. "
                "Try broadening your dataset selection or extending the SBERT range."
            )
        else:
            available_idx = filt.index.tolist()

            # Initialise or validate sample index
            if (
                "t3_sample_idx" not in st.session_state
                or st.session_state.t3_sample_idx not in available_idx
            ):
                st.session_state.t3_sample_idx = int(np.random.choice(available_idx))

            if st.button("Next Example →", key="t3_next"):
                st.session_state.t3_sample_idx = int(np.random.choice(available_idx))

            try:
                row = filt.loc[st.session_state.t3_sample_idx]
            except KeyError:
                st.session_state.t3_sample_idx = int(np.random.choice(available_idx))
                row = filt.loc[st.session_state.t3_sample_idx]

            # Metadata
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"**Dataset:** `{row.get('dataset', 'N/A')}`")
            m2.markdown(f"**Source Group:** `{row.get('src_group', 'N/A')}`")
            score = row.get(sbert_col, None)
            if score is not None and not pd.isna(score):
                score_color = (
                    "#4CAF50" if score > 0.85 else
                    "#FF9800" if score > 0.70 else "#F44336"
                )
                m3.markdown(
                    f"**SBERT Cosine (T0→T3):** "
                    f'<span style="color:{score_color};font-weight:700;">{score:.4f}</span>',
                    unsafe_allow_html=True,
                )

            tc1, sep, tc2 = st.columns([5, 1, 5])
            with tc1:
                st.markdown(
                    '<span style="color:#2196F3;font-weight:700;font-size:1.05rem;">T0 — Original</span>',
                    unsafe_allow_html=True,
                )
                t0_text = str(row.get("original", ""))[:3000] or "(empty)"
                st.text_area(
                    "t0", value=t0_text, height=280,
                    disabled=True, label_visibility="collapsed",
                )
            with sep:
                st.markdown("<br>" * 8, unsafe_allow_html=True)
                if score is not None and not pd.isna(score):
                    st.markdown(
                        f'<div style="text-align:center;">'
                        f'<span style="font-size:0.75rem;color:#888;">SBERT</span><br>'
                        f'<span style="font-size:1.5rem;font-weight:700;color:{score_color};">'
                        f'{score:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with tc2:
                st.markdown(
                    '<span style="color:#F44336;font-weight:700;font-size:1.05rem;">T3 — AI³ Paraphrase</span>',
                    unsafe_allow_html=True,
                )
                t3_text = str(row.get("chatgpt_chatgpt_chatgpt", ""))[:3000] or "(empty)"
                st.text_area(
                    "t3", value=t3_text, height=280,
                    disabled=True, label_visibility="collapsed",
                )

        st.divider()

        # ── t-SNE scatter ─────────────────────────────────────────────────────
        st.subheader("t-SNE Embedding Space")

        color_toggle = st.radio(
            "Colour by:", ["Iteration", "Source Group"],
            horizontal=True, key="t3_tsne_toggle",
        )

        # Sample for performance (8 k points is smooth in plotly)
        TSNE_N = 8000
        tsne_plot = tsne_df.sample(min(TSNE_N, len(tsne_df)), random_state=42)

        if sel_ds:
            tsne_plot = tsne_plot[tsne_plot["dataset"].isin(sel_ds)]

        if tsne_plot.empty:
            st.warning("No t-SNE points for the selected datasets.")
        else:
            hover_extra = [c for c in ["dataset", "src_group", "iteration"] if c in tsne_plot.columns]

            if color_toggle == "Iteration":
                fig_tsne = px.scatter(
                    tsne_plot, x="x", y="y", color="iteration",
                    color_discrete_map=ITER_COLORS,
                    category_orders={"iteration": ITER_ORDER},
                    title="t-SNE — Coloured by Paraphrase Iteration",
                    labels={"x": "t-SNE 1", "y": "t-SNE 2", "iteration": "Iteration"},
                    opacity=0.55, hover_data=hover_extra,
                )
            else:
                fig_tsne = px.scatter(
                    tsne_plot, x="x", y="y", color="src_group",
                    color_discrete_map=GROUP_COLORS,
                    title="t-SNE — Coloured by Source Group",
                    labels={"x": "t-SNE 1", "y": "t-SNE 2", "src_group": "Source Group"},
                    opacity=0.55, hover_data=hover_extra,
                )

            fig_tsne.update_traces(marker=dict(size=4))
            fig_tsne.update_layout(
                template="plotly_white", height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_tsne, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — POINT OF NO RETURN
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.header("Point of No Return — Authorship Attribution Decay")
        st.markdown(
            "*At what iteration does the model lose the ability to identify the original author?*"
        )
        st.divider()

        attr_df = load_attribution_f1()
        clf_results = load_clf_results()
        ds_f1 = load_ds_fingerprint_f1()
        ponr = summary.get("ponr_threshold", 0.55)

        # ── F1 decay line chart ───────────────────────────────────────────────
        fig_f1 = go.Figure()

        marker_colors = [ITER_COLORS.get(it, "#888") for it in attr_df["iteration"]]
        fig_f1.add_trace(go.Scatter(
            x=attr_df["iteration"], y=attr_df["f1"],
            mode="lines+markers",
            name="Attribution F1",
            line=dict(color="#2196F3", width=3),
            marker=dict(size=12, color=marker_colors,
                        line=dict(width=1.5, color="white")),
        ))
        fig_f1.add_hline(
            y=ponr, line_dash="dash", line_color="#F44336", line_width=2,
            annotation_text=f"Identity Loss Threshold  (F1 = {ponr})",
            annotation_position="top right",
            annotation_font=dict(color="#F44336", size=13),
        )

        # Annotate where identity is "lost" (first iter below threshold)
        below = attr_df[attr_df["f1"] < ponr]
        if not below.empty:
            row_b = below.iloc[0]
            fig_f1.add_annotation(
                x=row_b["iteration"], y=row_b["f1"],
                text=f"Identity Lost @ {row_b['iteration']}",
                showarrow=True, arrowhead=2, arrowcolor="#F44336",
                font=dict(color="#F44336", size=12), ax=60, ay=-40,
            )

        fig_f1.update_layout(
            title="Authorship Attribution F1 Across Paraphrase Iterations",
            xaxis_title="Iteration", yaxis_title="Macro F1 Score",
            yaxis=dict(range=[0.0, 1.0]),
            template="plotly_white", height=400,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        st.divider()

        col_clf, col_ds_fp = st.columns(2)

        # ── Classifier comparison bar chart ───────────────────────────────────
        with col_clf:
            clf_names = list(clf_results.keys())
            clf_means = [clf_results[k]["mean"] for k in clf_names]
            clf_stds = [clf_results[k]["std"] for k in clf_names]

            fig_clf = go.Figure(go.Bar(
                x=clf_names, y=clf_means,
                error_y=dict(type="data", array=clf_stds, visible=True, thickness=2, width=8),
                marker_color=["#2196F3", "#4CAF50"],
                text=[f"{v:.3f}" for v in clf_means],
                textposition="outside",
            ))
            fig_clf.update_layout(
                title="Paraphraser Fingerprinting — Classifier F1",
                xaxis_title="Classifier", yaxis_title="Macro F1 Score",
                yaxis=dict(range=[0.0, max(clf_means) * 1.3 if clf_means else 1.0]),
                template="plotly_white", height=380, showlegend=False,
            )
            st.plotly_chart(fig_clf, use_container_width=True)

        # ── Per-dataset F1 bar chart ──────────────────────────────────────────
        with col_ds_fp:
            valid_ds = {k: v for k, v in ds_f1.items() if v is not None}
            if valid_ds:
                ds_names = list(valid_ds.keys())
                ds_vals = list(valid_ds.values())
                fig_ds = px.bar(
                    x=ds_names, y=ds_vals,
                    title="Per-Dataset Fingerprinting F1 (Random Forest)",
                    labels={"x": "Dataset", "y": "Macro F1 Score"},
                    color=ds_vals, color_continuous_scale="Blues",
                    text=[f"{v:.3f}" for v in ds_vals],
                )
                fig_ds.update_traces(textposition="outside")
                fig_ds.update_layout(
                    yaxis=dict(range=[0.0, max(ds_vals) * 1.3]),
                    template="plotly_white", height=380,
                    showlegend=False, coloraxis_showscale=False,
                )
                st.plotly_chart(fig_ds, use_container_width=True)
            else:
                st.info("Per-dataset fingerprint F1 data unavailable.")

        st.divider()

        st.markdown(
            '<div class="forensic-box">'
            '<strong>Forensic Paradox:</strong> '
            '"If every linguistic marker is replaced by AI but meaning remains — <em>who is the author?</em>" '
            'Attribution classifiers lose the signal of the original author across iterations, '
            'yet each paraphraser imposes its own detectable fingerprint — '
            'suggesting the AI system itself becomes the de-facto author of the rewritten text.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — PARAPHRASER FINGERPRINTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab6:
        st.header("Paraphraser Fingerprints — Forensic Stylometry")
        st.markdown(
            "*Can we identify which AI paraphrased a text from its linguistic signature?*"
        )
        st.divider()

        fp_df = load_fingerprint_features()

        col_hist, col_feat = st.columns(2)

        # ── TTR overlapping histogram ─────────────────────────────────────────
        with col_hist:
            fig_ttr = px.histogram(
                fp_df, x="ttr", color="label",
                nbins=60, opacity=0.60, barmode="overlay",
                title="Type-Token Ratio Distribution by Paraphraser",
                labels={"ttr": "Type-Token Ratio (TTR)", "label": "Paraphraser"},
                color_discrete_sequence=PARAPHRASER_PALETTE,
            )
            fig_ttr.update_layout(
                template="plotly_white", height=400,
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_ttr, use_container_width=True)

        # ── Mean feature grouped bar chart ────────────────────────────────────
        with col_feat:
            feat_cols = [c for c in ["ttr", "avg_sent_len", "hapax_rate", "noun_rate", "verb_rate"]
                         if c in fp_df.columns]
            mean_feats = fp_df.groupby("label")[feat_cols].mean().reset_index()
            mean_long = mean_feats.melt(
                id_vars="label", var_name="Feature", value_name="Mean Value"
            )
            fig_feats = px.bar(
                mean_long, x="Feature", y="Mean Value", color="label",
                barmode="group",
                title="Mean Lexical Features per Paraphraser",
                labels={"Mean Value": "Mean Value", "Feature": "Feature", "label": "Paraphraser"},
                color_discrete_sequence=PARAPHRASER_PALETTE,
            )
            fig_feats.update_layout(
                template="plotly_white", height=400,
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_feats, use_container_width=True)

        st.divider()

        # ── Live Predictor ────────────────────────────────────────────────────
        st.subheader("Live Paraphraser Predictor")
        st.caption(
            "Paste any text to identify which AI paraphraser most likely produced it. "
            "**Note:** POS features (noun\\_rate, verb\\_rate, dep\\_depth) are set to 0 "
            "since spaCy is not available — predictions rely on lexical features only."
        )

        user_text = st.text_area(
            "Paste text here:", height=160,
            placeholder="Paste a paragraph of AI-paraphrased text to fingerprint…",
            key="pred_input",
        )

        if st.button("Identify Paraphraser", key="pred_btn") and user_text.strip():
            with st.spinner("Running model…"):
                try:
                    rf_model = load_rf_model()
                    scaler = load_scaler()
                    le = load_label_encoder()

                    feats = extract_text_features(user_text)

                    # Determine feature count from the fitted scaler
                    try:
                        n_feat = scaler.n_features_in_
                    except AttributeError:
                        n_feat = 8  # safe default

                    X = build_feature_vector(feats, n_feat)
                    X_scaled = scaler.transform(X)
                    probs = rf_model.predict_proba(X_scaled)[0]

                    # Resolve class labels
                    rf_classes = rf_model.classes_
                    try:
                        if np.issubdtype(np.array(rf_classes).dtype, np.integer):
                            class_labels = le.inverse_transform(rf_classes)
                        else:
                            class_labels = np.array(rf_classes, dtype=str)
                    except Exception:
                        class_labels = le.classes_ if hasattr(le, "classes_") else [
                            str(c) for c in rf_classes
                        ]

                    top3_idx = np.argsort(probs)[::-1][:3]
                    bar_palette = ["#2196F3", "#4CAF50", "#FF9800"]

                    st.markdown("#### Top-3 Predicted Paraphrasers")
                    for rank, idx in enumerate(top3_idx, 1):
                        label = class_labels[idx] if idx < len(class_labels) else f"Class {idx}"
                        prob = float(probs[idx])
                        col_lbl, col_bar, col_pct = st.columns([2, 6, 1])
                        with col_lbl:
                            st.markdown(
                                f'<span style="font-weight:700;color:{bar_palette[rank-1]};">'
                                f'{rank}. {label}</span>',
                                unsafe_allow_html=True,
                            )
                        with col_bar:
                            st.progress(prob)
                        with col_pct:
                            st.markdown(f"`{prob:.1%}`")

                    # Show extracted features for transparency
                    with st.expander("Extracted Feature Values"):
                        feat_display = {
                            k: round(v, 4) for k, v in feats.items()
                            if k not in ("n_sents", "n_words")
                        }
                        st.json(feat_display)

                except Exception as exc:
                    st.error(
                        f"Prediction failed: {exc}\n\n"
                        "Ensure rf_model.pkl, scaler.pkl, and label_encoder.pkl are present "
                        "in streamlit_assets/ and were trained on compatible features."
                    )


if __name__ == "__main__":
    main()
