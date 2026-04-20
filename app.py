"""
Ship of Theseus — Paraphrased Corpus
Research demo application.
Run: streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide", page_title="Ship of Theseus", page_icon="⚓")

ITER_COLORS  = {"T0": "#3B82F6", "T1": "#10B981", "T2": "#F59E0B", "T3": "#EF4444"}
ITER_ORDER   = ["T0", "T1", "T2", "T3"]
GROUP_COLORS = {"Human": "#3B82F6", "Synthetic": "#6366F1"}
PARA_PALETTE = px.colors.qualitative.Set2

ASSETS         = "streamlit_assets"
HF_REPO        = "deekshaSinha/ship-of-theseus-assets-model-results"
HF_LARGE_FILES = ["trajectory.csv"]


def _ensure_assets():
    os.makedirs(ASSETS, exist_ok=True)
    for fname in HF_LARGE_FILES:
        if not os.path.exists(f"{ASSETS}/{fname}"):
            with st.spinner("Preparing data files, please wait…"):
                hf_hub_download(
                    repo_id=HF_REPO, filename=fname,
                    repo_type="dataset", local_dir=ASSETS,
                )

_ensure_assets()


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
def load_dataset_stats() -> dict:
    path = f"{ASSETS}/dataset_stats.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
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
def load_dep_depth() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/dep_depth_decay.csv")

@st.cache_data
def load_feature_importances() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/feature_importances.csv", index_col=0)

@st.cache_data
def load_rq1_decay_summary() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/rq1_decay_summary.csv")

@st.cache_data
def load_fingerprint_features() -> pd.DataFrame:
    return pd.read_csv(f"{ASSETS}/fingerprint_features.csv")


def compute_drift_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"Iteration": "T0", "Mean": 1.0, "Std": 0.0}]
    for col in ["T1", "T2", "T3"]:
        if col in df.columns:
            vals = df[col].dropna()
            rows.append({"Iteration": col, "Mean": vals.mean(), "Std": vals.std()})
    return pd.DataFrame(rows)


def drift_chart(stats: pd.DataFrame, title: str, color: str, y_min: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stats["Iteration"], y=stats["Mean"] - stats["Std"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=stats["Iteration"], y=stats["Mean"] + stats["Std"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.12,)}",
        name="± 1 SD", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=stats["Iteration"], y=stats["Mean"],
        mode="lines+markers", name="Mean",
        line=dict(color=color, width=2.5), marker=dict(size=8, color=color),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1E293B")),
        xaxis_title="Iteration", yaxis_title="Cosine Similarity",
        yaxis=dict(range=[y_min, 1.03], gridcolor="#F1F5F9"),
        xaxis=dict(gridcolor="#F1F5F9"),
        template="plotly_white",
        legend=dict(orientation="h", y=1.1, font=dict(size=11)),
        height=360,
        margin=dict(l=50, r=20, t=55, b=45),
        plot_bgcolor="white",
    )
    return fig


CSS = """
<style>
[data-testid="stAppViewContainer"] > .main {background:#F8FAFC;}
.block-container {padding:2rem 3rem 3rem; max-width:1280px;}
h1 {font-size:1.75rem !important; font-weight:700 !important; color:#0F172A !important; letter-spacing:-0.01em;}
h2 {font-size:1.2rem !important; font-weight:600 !important; color:#1E293B !important;}
h3 {font-size:1rem !important; font-weight:600 !important; color:#1E293B !important;}
.stTabs [data-baseweb="tab-list"] {gap:0; border-bottom:2px solid #E2E8F0; background:transparent;}
.stTabs [data-baseweb="tab"] {
    padding:.55rem 1.1rem; font-size:.85rem; font-weight:500;
    color:#64748B; border-bottom:2px solid transparent; margin-bottom:-2px; background:transparent;
}
.stTabs [aria-selected="true"] {color:#2563EB !important; border-bottom-color:#2563EB !important;}
.kpi {background:white; border:1px solid #E2E8F0; border-radius:8px;
      padding:1.1rem 1.2rem; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,.04);}
.kpi-lbl {font-size:.68rem; color:#94A3B8; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem;}
.kpi-val {font-size:1.7rem; font-weight:700; color:#0F172A; line-height:1.1;}
.kpi-delta {font-size:.78rem; margin-top:.2rem;}
.ds-card {background:white; border:1px solid #E2E8F0; border-radius:8px;
          padding:1.1rem 1.25rem; margin-bottom:.75rem; box-shadow:0 1px 2px rgba(0,0,0,.03);}
.ds-title {font-size:.95rem; font-weight:600; color:#0F172A;}
.ds-meta {font-size:.75rem; color:#94A3B8; margin:3px 0 8px;}
.ds-body {font-size:.85rem; color:#475569; line-height:1.55; margin-bottom:10px;}
.stat-lbl {font-size:.67rem; color:#94A3B8; text-transform:uppercase; letter-spacing:.06em;}
.stat-val {font-size:.92rem; font-weight:600; color:#1E293B;}
.badge-human {background:#EFF6FF; color:#1D4ED8; padding:2px 9px;
              border-radius:99px; font-size:.75rem; font-weight:600;}
.badge-syn {background:#F5F3FF; color:#5B21B6; padding:2px 9px;
            border-radius:99px; font-size:.75rem; font-weight:600;}
.insight {background:#FFFBEB; border-left:3px solid #F59E0B; border-radius:0 6px 6px 0;
          padding:.85rem 1.1rem; font-size:.87rem; color:#1E293B; margin-top:1.25rem;}
.callout {background:#F0F9FF; border-left:3px solid #0EA5E9; border-radius:0 6px 6px 0;
          padding:.85rem 1.1rem; font-size:.87rem; color:#1E293B; margin-top:1.25rem;}
hr {border-color:#E2E8F0 !important;}
</style>
"""


def _render_schema_table(schema: list):
    header = (
        '<div style="background:white;border:1px solid #E2E8F0;border-radius:8px;'
        'overflow:hidden;margin:0.75rem 0 1.25rem;">'
        '<table style="width:100%;border-collapse:collapse;font-size:.84rem;">'
        '<thead><tr style="background:#F8FAFC;border-bottom:2px solid #E2E8F0;">'
        '<th style="padding:.6rem 1rem;text-align:left;color:#475569;font-weight:600;white-space:nowrap;">Column</th>'
        '<th style="padding:.6rem 1rem;text-align:left;color:#475569;font-weight:600;white-space:nowrap;">Type / Range</th>'
        '<th style="padding:.6rem 1rem;text-align:left;color:#475569;font-weight:600;">Possible Values</th>'
        '<th style="padding:.6rem 1rem;text-align:left;color:#475569;font-weight:600;">Description</th>'
        '</tr></thead><tbody>'
    )
    rows = ""
    for i, (col, typ, vals, desc) in enumerate(schema):
        bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFC"
        rows += (
            f'<tr style="background:{bg};border-bottom:1px solid #F1F5F9;">'
            f'<td style="padding:.5rem 1rem;font-family:monospace;color:#2563EB;white-space:nowrap;font-weight:600;">{col}</td>'
            f'<td style="padding:.5rem 1rem;white-space:nowrap;color:#64748B;">{typ}</td>'
            f'<td style="padding:.5rem 1rem;color:#374151;">{vals}</td>'
            f'<td style="padding:.5rem 1rem;color:#374151;">{desc}</td>'
            f'</tr>'
        )
    st.markdown(header + rows + "</tbody></table></div>", unsafe_allow_html=True)


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    summary = load_summary_stats()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Datasets",
        "Style Decay",
        "Drift Explorer",
        "Attribution",
        "Stylometry",
    ])

    # ── TAB 1 — OVERVIEW ────────────────────────────────────────────────────────
    with tab1:
        st.title("The Paradox of Authorial Identity")
        st.caption("Ship of Theseus Paraphrased Corpus  ·  NLP Research Defense  ·  April 2026")
        st.divider()

        n_docs   = summary.get("n_documents", 0)
        sbert_t3 = summary.get("mean_sbert_t3")
        f1_t0    = summary.get("attr_f1_t0")
        f1_t3    = summary.get("attr_f1_t3")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="kpi"><div class="kpi-lbl">Total Documents</div>'
                f'<div class="kpi-val">{n_docs:,}</div></div>', unsafe_allow_html=True)
        with c2:
            v = f"{sbert_t3:.4f}" if sbert_t3 is not None else "—"
            st.markdown(
                f'<div class="kpi"><div class="kpi-lbl">Mean SBERT Cosine @ T3</div>'
                f'<div class="kpi-val">{v}</div></div>', unsafe_allow_html=True)
        with c3:
            v = f"{f1_t0:.4f}" if f1_t0 is not None else "—"
            st.markdown(
                f'<div class="kpi"><div class="kpi-lbl">Attribution F1 @ T0</div>'
                f'<div class="kpi-val">{v}</div></div>', unsafe_allow_html=True)
        with c4:
            v = f"{f1_t3:.4f}" if f1_t3 is not None else "—"
            delta = ""
            if f1_t0 is not None and f1_t3 is not None:
                d = f1_t3 - f1_t0
                col = "#059669" if d >= 0 else "#DC2626"
                arrow = "▲" if d >= 0 else "▼"
                delta = f'<div class="kpi-delta" style="color:{col};">{arrow} {abs(d):.4f}</div>'
            st.markdown(
                f'<div class="kpi"><div class="kpi-lbl">Attribution F1 @ T3</div>'
                f'<div class="kpi-val">{v}</div>{delta}</div>', unsafe_allow_html=True)

        st.divider()

        datasets     = summary.get("datasets", [])
        paraphrasers = summary.get("paraphrasers", [])
        sbert_val    = sbert_t3 if sbert_t3 is not None else 0.87

        st.markdown("#### Research Summary")
        st.markdown(f"""
The **Ship of Theseus** thought experiment asks: if every plank of a ship is replaced one by one,
is it still the same ship? This research applies that paradox to **authorial identity in text**.
Across **{len(datasets)} corpora** (*{', '.join(datasets)}*) and **{len(paraphrasers)} paraphrasing
systems** (*{', '.join(paraphrasers)}*), documents are iteratively rewritten up to **three times**
(T0 → T1 → T2 → T3). The study tracks whether authorial style — measured via syntactic POS structure,
SBERT semantic embeddings, and lexical richness metrics (TTR, hapax rate) — survives progressive
paraphrasing. The central finding: semantics remain surprisingly resilient
(mean SBERT cosine ≈ **{sbert_val:.3f}** at T3), yet each paraphraser leaves forensically detectable
linguistic fingerprints — raising the question: *after three rounds of rewriting, whose identity
does the text ultimately bear?*
        """)

        st.markdown("")
        col_ds, col_par = st.columns(2)
        with col_ds:
            st.markdown("**Corpora:** " + "  ·  ".join(f"`{d}`" for d in datasets))
        with col_par:
            st.markdown("**Paraphrasers:** " + "  ·  ".join(f"`{p}`" for p in paraphrasers))

    # ── TAB 2 — DATASETS ────────────────────────────────────────────────────────
    with tab2:
        st.title("Dataset Explorer")
        st.markdown("Source corpora, feature schema, and corpus-level distributions.")
        st.divider()

        DATASET_INFO = {
            "cmv":     {"full_name": "Change My View (CMV)",              "source": "Reddit r/changemyview",          "domain": "Argumentative / Persuasion", "origin": "Human",
                        "description": "Long-form Reddit posts where users present an opinion and invite others to challenge it. Rich in structured argumentation, hedging language, and complex sentence constructions."},
            "eli5":    {"full_name": "Explain Like I'm 5 (ELI5)",         "source": "Reddit r/explainlikeimfive",     "domain": "Explanatory / Informal",     "origin": "Human",
                        "description": "Crowd-sourced simplified explanations of complex topics. Short sentences, informal register, and heavy use of analogy — contrasts sharply with academic corpora in lexical richness."},
            "sci_gen": {"full_name": "Scientific Abstracts (SciGen)",     "source": "arXiv / Academic Papers",        "domain": "Scientific / Technical",     "origin": "Synthetic",
                        "description": "Abstracts from scientific publications. Dense noun phrases, passive constructions, and high type-token ratios. Includes a synthetic subset to test stylometric distinguishability across origin types."},
            "tldr":    {"full_name": "TL;DR Summaries",                   "source": "Reddit (user-written)",          "domain": "Summary / Informal",         "origin": "Human",
                        "description": "User-generated 'too long; didn't read' summaries of Reddit posts. Extremely compressed and telegraphic — a low word-count edge case for stylometric analysis."},
            "wp":      {"full_name": "Writing Prompts (WP)",              "source": "Reddit r/WritingPrompts",        "domain": "Creative Fiction",           "origin": "Human",
                        "description": "Creative short stories written in response to community prompts. High narrative variety, broad vocabulary, and expressive stylistic choices — one of the richest corpora for authorship attribution."},
            "xsum":    {"full_name": "Extreme Summarisation (XSum)",      "source": "BBC News articles",              "domain": "News / Journalistic",        "origin": "Human",
                        "description": "Single-sentence summaries of BBC news articles. Formal journalistic register and consistent style conventions. Very short texts challenge paraphrasers to preserve meaning in minimal tokens."},
            "yelp":    {"full_name": "Yelp Reviews",                      "source": "Yelp Open Dataset",              "domain": "Review / Opinion",           "origin": "Human",
                        "description": "Consumer reviews spanning restaurants and businesses. Highly subjective, emotionally varied, and colloquial — sentiment-driven stylistic signals sensitive to paraphrasing."},
        }

        raw_stats = load_dataset_stats()

        # ── Corpus cards ─────────────────────────────────────────────────────
        for i in range(0, len(DATASET_INFO), 2):
            cols = st.columns(2)
            for j, (ds, info) in enumerate(list(DATASET_INFO.items())[i:i+2]):
                s = raw_stats.get(ds, {})
                n_docs  = s.get("unique_docs", "—")
                n_rows  = s.get("total_rows", "—")
                words   = s.get("mean_word_count", "—")
                badge_cls = "badge-syn" if info["origin"] == "Synthetic" else "badge-human"
                with cols[j]:
                    st.markdown(f"""
                    <div class="ds-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
                        <span class="ds-title">{info['full_name']}</span>
                        <span class="{badge_cls}">{info['origin']}</span>
                      </div>
                      <div class="ds-meta">{info['source']} &nbsp;·&nbsp; {info['domain']}</div>
                      <div class="ds-body">{info['description']}</div>
                      <div style="display:flex;gap:2rem;">
                        <div><div class="stat-lbl">Unique Docs</div><div class="stat-val">{f'{n_docs:,}' if isinstance(n_docs, int) else n_docs}</div></div>
                        <div><div class="stat-lbl">Total Rows</div><div class="stat-val">{f'{n_rows:,}' if isinstance(n_rows, int) else n_rows}</div></div>
                        <div><div class="stat-lbl">Mean Words</div><div class="stat-val">{words}</div></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Feature schema ────────────────────────────────────────────────────
        st.markdown("#### Data Schema")

        schema_tab1, schema_tab2 = st.tabs(["Raw Paraphrase Files  (corpus_paraphrased.csv)", "Feature Dataset  (fingerprint_features.csv)"])

        with schema_tab1:
            st.markdown(
                "One file per corpus (e.g. `cmv_paraphrased.csv`). Each row is one version of one document — "
                "original or paraphrased. Columns:"
            )
            RAW_SCHEMA = [
                ("source",       "categorical", "Human · OpenAI · PaLM · BigScience · Eleuther-AI · LLAMA · Tsinghua",
                 "Origin of the text — identifies the model or organisation that produced the original document"),
                ("key",          "string",      "e.g. cmv-258 · eli5-034 · xsum-112",
                 "Unique document identifier linking all versions (original + paraphrases) of the same text"),
                ("text",         "string",      "free text (variable length)",
                 "The actual text content — either the original document or one paraphrased version of it"),
                ("version_name", "categorical",
                 "original · chatgpt · chatgpt_chatgpt · chatgpt_chatgpt_chatgpt · palm · palm_palm · palm_palm_palm · "
                 "dipper(low) · dipper(low)_dipper(low) · dipper(low)_dipper(low)_dipper(low) · "
                 "dipper(high) · pegasus(slight) · pegasus(full) · …",
                 "Encodes which paraphraser was applied and how many times. "
                 "Single name = T1, doubled = T2, tripled = T3. 'original' = T0 baseline."),
            ]
            _render_schema_table(RAW_SCHEMA)

        with schema_tab2:
            st.markdown(
                "Each record represents one document version with its extracted stylometric features. "
                "Used for fingerprinting and classification. Columns:"
            )
            FEAT_SCHEMA = [
                ("dataset",      "categorical", "cmv · eli5 · sci_gen · tldr · wp · xsum · yelp",              "Source corpus the document belongs to"),
                ("label",        "categorical", "chatgpt · palm · dipper_low · dipper_bare · pegasus_slight · pegasus_full · dipper_high", "Paraphrasing system that produced this version"),
                ("ttr",          "float  [0, 1]",  "0.0 – 1.0",         "Type-Token Ratio — unique words ÷ total words; higher = more lexical diversity"),
                ("hapax_rate",   "float  [0, 1]",  "0.0 – 1.0",         "Proportion of words appearing exactly once; proxy for vocabulary novelty"),
                ("avg_sent_len", "float  ≥ 0",     "typically 5 – 40",  "Mean number of words per sentence"),
                ("avg_word_len", "float  ≥ 0",     "typically 4 – 7",   "Mean number of characters per word"),
                ("punct_rate",   "float  [0, 1]",  "0.0 – 0.10",        "Punctuation density — punctuation characters ÷ total characters"),
                ("n_words",      "float  ≥ 0",     "1 – 5,000+",        "Total word count of the document"),
                ("n_sents",      "float  ≥ 0",     "1 – 500+",          "Total sentence count of the document"),
                ("noun_rate",    "float  [0, 1]",  "0.0 – 1.0",         "Proportion of tokens tagged as nouns (POS)"),
                ("verb_rate",    "float  [0, 1]",  "0.0 – 1.0",         "Proportion of tokens tagged as verbs (POS)"),
                ("dep_depth",    "float  ≥ 0",     "typically 2 – 12",  "Mean dependency parse tree depth; higher = more syntactic nesting"),
            ]
            _render_schema_table(FEAT_SCHEMA)

        st.divider()

        # ── Two focused charts (only when raw_stats available) ────────────────
        if raw_stats:
            chart_df = pd.DataFrame([
                {"corpus": ds, "unique_docs": v["unique_docs"], "total_rows": v["total_rows"],
                 "mean_words": v["mean_word_count"]}
                for ds, v in raw_stats.items()
            ])

            ca, cb = st.columns(2)
            with ca:
                fig = px.bar(
                    chart_df.sort_values("unique_docs", ascending=False),
                    x="corpus", y="unique_docs",
                    title="Unique Documents per Corpus",
                    labels={"corpus": "Corpus", "unique_docs": "Unique Documents"},
                    color="unique_docs", color_continuous_scale="Blues", text="unique_docs",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(template="plotly_white", height=360,
                                  coloraxis_showscale=False, showlegend=False,
                                  margin=dict(t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

            with cb:
                src_rows = []
                for ds, v in raw_stats.items():
                    for src, cnt in v.get("sources", {}).items():
                        src_rows.append({"corpus": ds, "source": src, "count": cnt})
                if src_rows:
                    src_df = pd.DataFrame(src_rows)
                    fig = px.bar(
                        src_df, x="corpus", y="count", color="source", barmode="stack",
                        title="Text Source Distribution per Corpus",
                        labels={"corpus": "Corpus", "count": "Rows", "source": "Source"},
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    )
                    fig.update_layout(template="plotly_white", height=360,
                                      legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                                      margin=dict(t=50, b=80))
                    st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3 — STYLE DECAY ─────────────────────────────────────────────────────
    with tab3:
        st.title("Style Decay")
        st.markdown("Tracking syntactic structure and semantic similarity across paraphrase iterations.")
        st.divider()

        view_mode = st.radio(
            "View mode",
            ["Individual (per paraphraser)", "Aggregate (mean ± std)"],
            horizontal=True, key="decay_view",
        )
        aggregate = view_mode.startswith("Aggregate")

        pos_df   = load_pos_cosine_drift()
        sbert_df = load_sbert_cosine_drift()

        c1, c2 = st.columns(2)
        if aggregate:
            pos_stats   = compute_drift_stats(pos_df)
            sbert_stats = compute_drift_stats(sbert_df)
            with c1:
                st.plotly_chart(
                    drift_chart(pos_stats, "Syntactic Structure Stability (POS Cosine) — Mean ± SD", "#3B82F6", 0.85),
                    use_container_width=True)
            with c2:
                st.plotly_chart(
                    drift_chart(sbert_stats, "Semantic Identity Drift (SBERT Cosine) — Mean ± SD", "#EF4444", 0.55),
                    use_container_width=True)
        else:
            iter_order = ["T0", "T1", "T2", "T3"]
            # Melt wide CSVs (paraphraser, T1, T2, T3) → long, prepend T0=1.0 per paraphraser
            def _to_long(df: pd.DataFrame, label: str) -> pd.DataFrame:
                df = df.copy()
                df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
                id_col = df.columns[0]
                t_cols = [c for c in ["T1", "T2", "T3"] if c in df.columns]
                long   = df.melt(id_vars=id_col, value_vars=t_cols,
                                 var_name="iteration", value_name=label)
                long   = long.rename(columns={id_col: "paraphraser"})
                t0     = pd.DataFrame({"paraphraser": df[id_col], "iteration": "T0", label: 1.0})
                return pd.concat([t0, long], ignore_index=True)

            pos_long   = _to_long(pos_df,   "POS Cosine")
            sbert_long = _to_long(sbert_df, "SBERT Cosine")
            for df_long in [pos_long, sbert_long]:
                df_long["iteration"] = pd.Categorical(df_long["iteration"],
                                                      categories=iter_order, ordered=True)

            with c1:
                fig = px.line(
                    pos_long.sort_values("iteration"),
                    x="iteration", y="POS Cosine", color="paraphraser", markers=True,
                    title="Syntactic Structure Stability (POS Cosine) — per Paraphraser",
                    labels={"iteration": "Iteration", "POS Cosine": "Cosine Similarity",
                            "paraphraser": "Paraphraser"},
                    color_discrete_sequence=PARA_PALETTE,
                )
                fig.update_layout(template="plotly_white", height=370,
                                  legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                                  margin=dict(t=55, b=80))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.line(
                    sbert_long.sort_values("iteration"),
                    x="iteration", y="SBERT Cosine", color="paraphraser", markers=True,
                    title="Semantic Identity Drift (SBERT Cosine) — per Paraphraser",
                    labels={"iteration": "Iteration", "SBERT Cosine": "Cosine Similarity",
                            "paraphraser": "Paraphraser"},
                    color_discrete_sequence=PARA_PALETTE,
                )
                fig.update_layout(template="plotly_white", height=370,
                                  legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                                  margin=dict(t=55, b=80))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        ling_df = load_linguistic_delta()
        if "iteration" in ling_df.columns:
            ling_df["iteration"] = pd.Categorical(ling_df["iteration"], categories=ITER_ORDER, ordered=True)
            ling_df = ling_df.sort_values(["paraphraser", "iteration"])

        c3, c4 = st.columns(2)
        if aggregate:
            agg3 = ling_df.groupby("iteration", observed=True)[["RTTR", "N Words"]].agg(["mean","std"]).reset_index()
            with c3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=agg3["iteration"],
                    y=agg3[("RTTR","mean")] - agg3[("RTTR","std")],
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=agg3["iteration"],
                    y=agg3[("RTTR","mean")] + agg3[("RTTR","std")],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(59,130,246,0.12)", name="± 1 SD", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=agg3["iteration"], y=agg3[("RTTR","mean")],
                    mode="lines+markers", name="Mean RTTR",
                    line=dict(color="#3B82F6", width=2.5), marker=dict(size=8)))
                fig.update_layout(title="Root Type-Token Ratio — Mean ± SD",
                    xaxis_title="Iteration", yaxis_title="RTTR",
                    template="plotly_white", height=350,
                    legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
                    margin=dict(t=50, b=80))
                st.plotly_chart(fig, use_container_width=True)

            with c4:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=agg3["iteration"],
                    y=agg3[("N Words","mean")] - agg3[("N Words","std")],
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=agg3["iteration"],
                    y=agg3[("N Words","mean")] + agg3[("N Words","std")],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(16,185,129,0.12)", name="± 1 SD", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=agg3["iteration"], y=agg3[("N Words","mean")],
                    mode="lines+markers", name="Mean Word Count",
                    line=dict(color="#10B981", width=2.5), marker=dict(size=8)))
                fig.update_layout(title="Mean Word Count — Mean ± SD",
                    xaxis_title="Iteration", yaxis_title="Word Count",
                    template="plotly_white", height=350,
                    legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
                    margin=dict(t=50, b=80))
                st.plotly_chart(fig, use_container_width=True)
        else:
            with c3:
                fig = px.line(
                    ling_df, x="iteration", y="RTTR", color="paraphraser", markers=True,
                    title="Root Type-Token Ratio per Paraphraser (T0 → T3)",
                    labels={"RTTR": "RTTR", "iteration": "Iteration", "paraphraser": "Paraphraser"},
                    color_discrete_sequence=PARA_PALETTE,
                )
                fig.update_layout(template="plotly_white", height=350,
                                  legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                                  margin=dict(t=50, b=80))
                st.plotly_chart(fig, use_container_width=True)

            with c4:
                fig = px.line(
                    ling_df, x="iteration", y="N Words", color="paraphraser", markers=True,
                    title="Mean Word Count per Paraphraser (T0 → T3)",
                    labels={"N Words": "Word Count", "iteration": "Iteration", "paraphraser": "Paraphraser"},
                    color_discrete_sequence=PARA_PALETTE,
                )
                fig.update_layout(template="plotly_white", height=350,
                                  legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                                  margin=dict(t=50, b=80))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Multi-Modal Decay — Normalized to T0")
        dep_df  = load_dep_depth()
        norm_rows = []
        for para, grp in ling_df.groupby("paraphraser"):
            t0_rttr = grp.loc[grp["iteration"] == "T0", "RTTR"].values
            if not len(t0_rttr):
                continue
            t0_rttr = t0_rttr[0]
            dep_grp = dep_df[dep_df["paraphraser"] == para]
            t0_dep  = dep_grp.loc[dep_grp["iteration"] == "T0", "dep_depth"].values
            if not len(t0_dep):
                continue
            t0_dep = t0_dep[0]
            for _, row in grp.iterrows():
                dep_row = dep_grp[dep_grp["iteration"] == row["iteration"]]
                dep_val = dep_row["dep_depth"].values[0] if len(dep_row) else None
                if dep_val is None:
                    continue
                norm_rows.append({
                    "Iteration": row["iteration"],
                    "Structural (Dep-Depth)": dep_val / t0_dep if t0_dep else 1.0,
                    "Lexical (RTTR)": row["RTTR"] / t0_rttr if t0_rttr else 1.0,
                })
        norm_df   = pd.DataFrame(norm_rows).groupby("Iteration")[["Structural (Dep-Depth)", "Lexical (RTTR)"]].mean().reset_index()
        norm_long = norm_df.melt(id_vars="Iteration", var_name="Metric", value_name="Normalized Value")
        fig_norm  = px.line(
            norm_long, x="Iteration", y="Normalized Value", color="Metric", markers=True,
            title="Structural vs. Lexical Decay (Normalized to T0 = 1.0)",
            labels={"Iteration": "Iteration", "Normalized Value": "Relative to T0", "Metric": "Dimension"},
            color_discrete_map={"Structural (Dep-Depth)": "#3B82F6", "Lexical (RTTR)": "#F59E0B"},
        )
        fig_norm.add_hline(y=1.0, line_dash="dot", line_color="#94A3B8", line_width=1,
                           annotation_text="T0 baseline", annotation_position="top left",
                           annotation_font=dict(color="#94A3B8", size=11))
        fig_norm.update_traces(line=dict(width=2.5), marker=dict(size=9))
        fig_norm.update_layout(template="plotly_white", height=380,
                               legend=dict(orientation="h", y=-0.25, font=dict(size=12)),
                               margin=dict(t=55, b=80))
        st.plotly_chart(fig_norm, use_container_width=True)

        st.markdown(
            '<div class="insight"><strong>Key Finding —</strong> '
            'The syntactic skeleton (dep-depth) remains closer to 1.0 than the lexical skin (RTTR), '
            'confirming that grammatical structure outlasts vocabulary under iterative paraphrasing.</div>',
            unsafe_allow_html=True)

    # ── TAB 4 — DRIFT EXPLORER ──────────────────────────────────────────────────
    with tab4:
        st.title("Drift Explorer")
        st.markdown("Browse how original texts (T0) transform through three rounds of paraphrasing (T3).")
        st.divider()

        trajectory_df = load_trajectory()
        tsne_df       = load_tsne_coords()
        sbert_col     = "sbert_cos_t3"

        fc1, fc2, fc3 = st.columns([2, 1, 2])
        with fc1:
            all_ds  = sorted(trajectory_df["dataset"].dropna().unique().tolist())
            sel_ds  = st.multiselect("Corpus", options=all_ds, default=all_ds, key="ex_ds")
        with fc2:
            all_grp = sorted(trajectory_df["src_group"].dropna().unique().tolist())
            sel_grp = st.selectbox("Source Group", options=["All"] + all_grp, index=0, key="ex_grp")
        with fc3:
            valid_sbert = trajectory_df[sbert_col].dropna() if sbert_col in trajectory_df.columns else pd.Series([0.0, 1.0])
            s_min, s_max = float(valid_sbert.min()), float(valid_sbert.max())
            sbert_range = st.slider("SBERT Cosine @ T3",
                min_value=round(s_min, 2), max_value=round(s_max, 2),
                value=(round(s_min, 2), round(s_max, 2)), step=0.01, key="ex_sbert")

        filt = trajectory_df.copy()
        if sel_ds:
            filt = filt[filt["dataset"].isin(sel_ds)]
        if sel_grp != "All" and sel_grp in filt["src_group"].values:
            filt = filt[filt["src_group"] == sel_grp]
        if sbert_col in filt.columns:
            filt = filt[filt[sbert_col].between(sbert_range[0], sbert_range[1], inclusive="both")]

        st.divider()
        st.subheader("Original vs. Paraphrase (T0 / T3)")

        if filt.empty:
            st.warning("No records match the current filters. Broaden the selection or SBERT range.")
        else:
            available_idx = filt.index.tolist()
            if "ex_idx" not in st.session_state or st.session_state.ex_idx not in available_idx:
                st.session_state.ex_idx = int(np.random.choice(available_idx))
            if st.button("Next sample →", key="ex_next"):
                st.session_state.ex_idx = int(np.random.choice(available_idx))
            try:
                row = filt.loc[st.session_state.ex_idx]
            except KeyError:
                st.session_state.ex_idx = int(np.random.choice(available_idx))
                row = filt.loc[st.session_state.ex_idx]

            m1, m2, m3 = st.columns(3)
            m1.markdown(f"**Corpus:** `{row.get('dataset', '—')}`")
            m2.markdown(f"**Source Group:** `{row.get('src_group', '—')}`")
            score = row.get(sbert_col)
            if score is not None and not pd.isna(score):
                sc = "#10B981" if score > 0.85 else "#F59E0B" if score > 0.70 else "#EF4444"
                m3.markdown(
                    f"**SBERT Cosine (T0 → T3):** "
                    f'<span style="color:{sc};font-weight:700;">{score:.4f}</span>',
                    unsafe_allow_html=True)

            tc1, sep, tc2 = st.columns([5, 1, 5])
            with tc1:
                st.markdown('<span style="font-weight:600;color:#3B82F6;">T0 — Original</span>',
                            unsafe_allow_html=True)
                st.text_area("t0", value=str(row.get("original", ""))[:3000] or "(empty)",
                             height=280, disabled=True, label_visibility="collapsed")
            with sep:
                st.markdown("<br>" * 8, unsafe_allow_html=True)
                if score is not None and not pd.isna(score):
                    st.markdown(
                        f'<div style="text-align:center;">'
                        f'<span style="font-size:.7rem;color:#94A3B8;">SBERT</span><br>'
                        f'<span style="font-size:1.4rem;font-weight:700;color:{sc};">{score:.3f}</span>'
                        f'</div>', unsafe_allow_html=True)
            with tc2:
                st.markdown('<span style="font-weight:600;color:#EF4444;">T3 — Paraphrase</span>',
                            unsafe_allow_html=True)
                st.text_area("t3", value=str(row.get("chatgpt_chatgpt_chatgpt", ""))[:3000] or "(empty)",
                             height=280, disabled=True, label_visibility="collapsed")

        st.divider()
        st.subheader("Embedding Space (t-SNE)")

        color_by = st.radio("Colour by:", ["Iteration", "Source Group"],
                            horizontal=True, key="ex_tsne_color")

        tsne_plot = tsne_df.sample(min(8000, len(tsne_df)), random_state=42)
        if sel_ds:
            tsne_plot = tsne_plot[tsne_plot["dataset"].isin(sel_ds)]

        if tsne_plot.empty:
            st.warning("No embedding points for the selected corpora.")
        else:
            hover_cols = [c for c in ["dataset", "src_group", "iteration"] if c in tsne_plot.columns]
            if color_by == "Iteration":
                fig = px.scatter(tsne_plot, x="x", y="y", color="iteration",
                    color_discrete_map=ITER_COLORS, category_orders={"iteration": ITER_ORDER},
                    title="t-SNE — by Paraphrase Iteration",
                    labels={"x": "t-SNE 1", "y": "t-SNE 2"}, opacity=0.5, hover_data=hover_cols)
            else:
                fig = px.scatter(tsne_plot, x="x", y="y", color="src_group",
                    color_discrete_map=GROUP_COLORS,
                    title="t-SNE — by Source Group",
                    labels={"x": "t-SNE 1", "y": "t-SNE 2"}, opacity=0.5, hover_data=hover_cols)
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(template="plotly_white", height=480,
                              legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
                              margin=dict(t=55, b=40))
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5 — ATTRIBUTION ─────────────────────────────────────────────────────
    with tab5:
        st.title("Authorship Attribution Decay")
        st.markdown("At what iteration does attribution performance drop below a meaningful threshold?")
        st.divider()

        attr_wide   = load_attribution_f1()
        clf_results = load_clf_results()
        ds_f1       = load_ds_fingerprint_f1()
        ponr        = summary.get("ponr_threshold", 0.55)

        # Normalise column names (strip BOM / whitespace that Cloud CSV readers may add)
        attr_wide = attr_wide.copy()
        attr_wide.columns = [c.strip().lstrip("\ufeff") for c in attr_wide.columns]
        iter_cols = [c for c in ["T0", "T1", "T2", "T3"] if c in attr_wide.columns]

        if iter_cols:
            # Wide format: paraphraser | T0 | T1 | T2 | T3
            id_col    = next((c for c in attr_wide.columns if c not in iter_cols), attr_wide.columns[0])
            attr_long = attr_wide.melt(id_vars=id_col, value_vars=iter_cols,
                                       var_name="iteration", value_name="f1")
            attr_long = attr_long.rename(columns={id_col: "paraphraser"})
        else:
            # Legacy format: iteration | f1
            attr_long = attr_wide.rename(columns={attr_wide.columns[0]: "iteration",
                                                   attr_wide.columns[1]: "f1"})
            attr_long["paraphraser"] = "mean"
            iter_cols = [c for c in ["T0", "T1", "T2", "T3"] if c in attr_long["iteration"].values]

        attr_mean = attr_long.groupby("iteration", sort=False)["f1"].mean().reindex(iter_cols).reset_index()
        attr_mean.columns = ["iteration", "f1"]

        fig_f1 = go.Figure()
        # Per-paraphraser light traces
        for para, grp in attr_long.groupby("paraphraser"):
            grp = grp.set_index("iteration").reindex(iter_cols).reset_index()
            fig_f1.add_trace(go.Scatter(
                x=grp["iteration"], y=grp["f1"],
                mode="lines", name=para, opacity=0.35,
                line=dict(width=1.2, dash="dot"),
                hovertemplate=f"{para}: %{{y:.3f}}<extra></extra>",
            ))
        # Mean line
        marker_colors = [ITER_COLORS.get(it, "#888") for it in attr_mean["iteration"]]
        fig_f1.add_trace(go.Scatter(
            x=attr_mean["iteration"], y=attr_mean["f1"],
            mode="lines+markers", name="Mean F1",
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=11, color=marker_colors, line=dict(width=1.5, color="white")),
        ))
        fig_f1.add_hline(y=ponr, line_dash="dash", line_color="#EF4444", line_width=1.5,
            annotation_text=f"Threshold  (F1 = {ponr})",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=12))
        below_mean = attr_mean[attr_mean["f1"] < ponr]
        if not below_mean.empty:
            rb = below_mean.iloc[0]
            fig_f1.add_annotation(
                x=rb["iteration"], y=rb["f1"],
                text=f"Mean below threshold @ {rb['iteration']}",
                showarrow=True, arrowhead=2, arrowcolor="#EF4444",
                font=dict(color="#EF4444", size=11), ax=60, ay=-40)
        fig_f1.update_layout(
            title="Authorship Attribution F1 Across Iterations",
            xaxis_title="Iteration", yaxis_title="Macro F1",
            yaxis=dict(range=[0.0, 1.0], gridcolor="#F1F5F9"),
            xaxis=dict(gridcolor="#F1F5F9"),
            template="plotly_white", height=420,
            legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
            margin=dict(t=55, b=80))
        st.plotly_chart(fig_f1, use_container_width=True)

        st.divider()
        ca, cb = st.columns(2)

        with ca:
            clf_names = list(clf_results.keys())
            clf_means = [clf_results[k]["mean"] for k in clf_names]
            clf_stds  = [clf_results[k]["std"]  for k in clf_names]
            fig = go.Figure(go.Bar(
                x=clf_names, y=clf_means,
                error_y=dict(type="data", array=clf_stds, visible=True, thickness=1.5, width=6),
                marker_color=["#3B82F6", "#10B981"],
                text=[f"{v:.3f}" for v in clf_means], textposition="outside",
            ))
            fig.update_layout(
                title="Fingerprinting Classifier Comparison (F1)",
                xaxis_title="Classifier", yaxis_title="Macro F1",
                yaxis=dict(range=[0.0, max(clf_means) * 1.35 if clf_means else 1.0],
                           gridcolor="#F1F5F9"),
                template="plotly_white", height=370,
                showlegend=False, margin=dict(t=50, b=45))
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            valid_ds = {k: v for k, v in ds_f1.items() if v is not None}
            if valid_ds:
                fig = px.bar(
                    x=list(valid_ds.keys()), y=list(valid_ds.values()),
                    title="Per-Corpus Fingerprinting F1 (Random Forest)",
                    labels={"x": "Corpus", "y": "Macro F1"},
                    color=list(valid_ds.values()), color_continuous_scale="Blues",
                    text=[f"{v:.3f}" for v in valid_ds.values()],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    yaxis=dict(range=[0.0, max(valid_ds.values()) * 1.35],
                               gridcolor="#F1F5F9"),
                    template="plotly_white", height=370,
                    showlegend=False, coloraxis_showscale=False, margin=dict(t=50, b=45))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Per-corpus F1 data unavailable.")

        st.markdown(
            '<div class="callout"><strong>Finding —</strong> '
            'Attribution F1 never drops below the 0.55 threshold at any iteration — there is no Point of No Return. '
            'Instead, a Point of Maximum Contrast emerges at T1 as multi-source paraphraser diversity collapses into a '
            'single style, making human-authored texts paradoxically more identifiable. Each paraphrasing system also '
            'leaves its own detectable stylometric fingerprint, identifiable from surface features alone at '
            '2.6× above chance.</div>',
            unsafe_allow_html=True)

    # ── TAB 6 — STYLOMETRY ──────────────────────────────────────────────────────
    with tab6:
        st.title("Forensic Stylometry")
        st.markdown("Do different paraphrasing systems leave distinct, detectable stylistic signatures?")
        st.divider()

        fp_df     = load_fingerprint_features()
        feat_cols = [c for c in ["ttr", "avg_sent_len", "hapax_rate", "noun_rate", "verb_rate"]
                     if c in fp_df.columns]

        ca, cb = st.columns(2)
        with ca:
            fig = px.histogram(
                fp_df, x="ttr", color="label",
                nbins=60, opacity=0.55, barmode="overlay",
                title="Type-Token Ratio Distribution by Paraphraser",
                labels={"ttr": "Type-Token Ratio (TTR)", "label": "Paraphraser"},
                color_discrete_sequence=PARA_PALETTE,
            )
            fig.update_layout(template="plotly_white", height=380,
                              legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                              margin=dict(t=50, b=80))
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            mean_feats = fp_df.groupby("label")[feat_cols].mean().reset_index()
            mean_long  = mean_feats.melt(id_vars="label", var_name="Feature", value_name="Mean Value")
            fig = px.bar(
                mean_long, x="Feature", y="Mean Value", color="label", barmode="group",
                title="Mean Lexical Features per Paraphraser",
                labels={"Mean Value": "Mean Value", "label": "Paraphraser"},
                color_discrete_sequence=PARA_PALETTE,
            )
            fig.update_layout(template="plotly_white", height=380,
                              legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                              margin=dict(t=50, b=80))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        cc, cd = st.columns(2)
        with cc:
            fig = px.box(
                fp_df, x="label", y="avg_sent_len",
                title="Sentence Length Distribution by Paraphraser",
                labels={"label": "Paraphraser", "avg_sent_len": "Avg Sentence Length (words)"},
                color="label", color_discrete_sequence=PARA_PALETTE,
            )
            fig.update_layout(template="plotly_white", height=380,
                              showlegend=False, margin=dict(t=50, b=45))
            st.plotly_chart(fig, use_container_width=True)

        with cd:
            fig = px.box(
                fp_df, x="label", y="hapax_rate",
                title="Hapax Rate Distribution by Paraphraser",
                labels={"label": "Paraphraser", "hapax_rate": "Hapax Rate"},
                color="label", color_discrete_sequence=PARA_PALETTE,
            )
            fig.update_layout(template="plotly_white", height=380,
                              showlegend=False, margin=dict(t=50, b=45))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight"><strong>Finding —</strong> '
            'Despite operating on the same source texts, each paraphrasing system produces '
            'statistically distinct distributions in TTR, sentence length, and hapax rate — '
            'confirming that stylometric fingerprinting can reliably identify the rewriting system '
            'even when surface semantics are preserved.</div>',
            unsafe_allow_html=True)


if __name__ == "__main__":
    main()
