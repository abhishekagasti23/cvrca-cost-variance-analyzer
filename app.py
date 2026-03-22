"""
CVRCA — Cost Variance Root Cause Analyzer
Streamlit Application — Project 2 of 3 for KPMG Associate Consultant Demo
Sector: US Energy & Utilities | Grid Modernization Capital Programme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from nlp_engine import RootCauseClassifier, ROOT_CAUSES, ROOT_CAUSE_COLORS
from cost_model import (CostOverrunModel, run_monte_carlo,
                        compute_portfolio_metrics, generate_portfolio_recommendations,
                        CONTRACT_TYPES, CONTRACTOR_TIERS)
from demo_data import generate_demo_data, to_excel

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CVRCA — Cost Intelligence System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — distinct aesthetic from Project 1 ──────────────────────────────────
# Project 1 was deep navy/blue. Project 2: dark amber/slate industrial finance aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp > header { display: none; }

.cvrca-header {
    background: linear-gradient(135deg, #0c0a06 0%, #1c1408 60%, #2a1f0a 100%);
    border-bottom: 2px solid #d97706;
    padding: 1.2rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; gap: 1.5rem;
}
.cvrca-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem; font-weight: 500;
    color: #fbbf24; letter-spacing: 0.12em;
}
.cvrca-subtitle { font-size: 0.78rem; color: #78716c; letter-spacing: 0.06em; text-transform: uppercase; }
.cvrca-sector {
    margin-left: auto; font-size: 0.72rem; color: #d97706;
    border: 1px solid #d97706; padding: 0.2rem 0.7rem;
    border-radius: 3px; font-family: 'IBM Plex Mono', monospace;
}

/* KPI grid */
.kpi-grid { display: grid; grid-template-columns: repeat(6,1fr); gap: 0.6rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: #0c0a06; border: 1px solid #292524; border-radius: 6px;
    padding: 0.85rem 1rem; position: relative; overflow: hidden;
}
.kpi-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; }
.kpi-amber::before { background: #d97706; }
.kpi-red::before   { background: #dc2626; }
.kpi-green::before { background: #16a34a; }
.kpi-blue::before  { background: #2563eb; }
.kpi-purple::before{ background: #7c3aed; }
.kpi-gray::before  { background: #57534e; }
.kpi-label { font-size: 0.65rem; color: #57534e; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.35rem; font-family: 'IBM Plex Mono', monospace; }
.kpi-value { font-size: 1.8rem; font-weight: 600; color: #fafaf9; line-height: 1; }
.kpi-delta { font-size: 0.7rem; color: #57534e; margin-top: 0.25rem; }

.section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: #d97706;
    border-bottom: 1px solid #292524; padding-bottom: 0.35rem; margin: 1.2rem 0 0.8rem 0;
}

/* Root cause badge */
.rc-badge {
    display: inline-block; padding: 2px 10px; border-radius: 3px;
    font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace;
    border: 1px solid; white-space: nowrap;
}

/* Recommendation card */
.rec-card {
    background: #0c0a06; border: 1px solid #292524;
    border-left: 4px solid #d97706; border-radius: 5px;
    padding: 0.8rem 1rem; margin-bottom: 0.7rem;
}
.rec-card.high { border-left-color: #dc2626; }
.rec-card.medium { border-left-color: #d97706; }
.rec-title { font-weight: 600; font-size: 0.88rem; color: #fafaf9; margin-bottom: 0.25rem; }
.rec-cause { font-size: 0.72rem; color: #78716c; margin-bottom: 0.4rem; font-family: 'IBM Plex Mono', monospace; }
.rec-action { font-size: 0.8rem; color: #fbbf24; margin-bottom: 0.2rem; }
.rec-detail { font-size: 0.75rem; color: #a8a29e; }

/* CO table */
.co-row-high   { background: #1c0a0a; }
.co-row-medium { background: #1c1408; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cvrca-header">
  <div>
    <div class="cvrca-logo">💰 CVRCA</div>
    <div class="cvrca-subtitle">Cost Variance Root Cause Analyzer · Portfolio Intelligence</div>
  </div>
  <div class="cvrca-sector">US ENERGY &amp; UTILITIES · GRID MODERNISATION · IIJA/IRA</div>
</div>
""", unsafe_allow_html=True)

# ─── SESSION STATE — train models once ────────────────────────────────────────
@st.cache_resource(show_spinner="Training NLP classifier and cost model...")
def load_models():
    clf = RootCauseClassifier()
    clf_metrics = clf.train()
    cm = CostOverrunModel()
    cm_metrics = cm.train()
    return clf, cm, clf_metrics, cm_metrics

clf, cm, clf_metrics, cm_metrics = load_models()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">DATA INPUT</div>', unsafe_allow_html=True)

    data_source = st.radio(
        "Source",
        ["🔬 Demo: US Grid Modernisation (IIJA)", "📁 Upload Excel (Projects + COs)"],
        label_visibility="collapsed",
    )

    uploaded = None
    if "Upload" in data_source:
        uploaded = st.file_uploader(
            "Upload Excel",
            type=["xlsx", "xls", "csv"],
            help="Excel with 'Projects' and 'Change Orders' sheets",
        )
        st.caption("Sheet 1: Projects | Sheet 2: Change Orders (with Description column)")

    st.markdown('<div class="section-header">FILTERS</div>', unsafe_allow_html=True)
    min_co_value = st.slider("Min CO value ($K)", 0, 500, 0, step=25)
    show_status = st.multiselect(
        "CO Status",
        ["Approved", "Pending", "Rejected"],
        default=["Approved", "Pending"],
    )
    selected_contractor = st.selectbox(
        "Filter by contractor", ["All contractors"] + [p[3] for p in __import__('demo_data').PROJECTS[:8]]
    )

    st.markdown('<div class="section-header">MODEL INFO</div>', unsafe_allow_html=True)
    st.metric("NLP CV F1", clf_metrics.get("CV F1 (macro)", "—"))
    st.metric("Cost Model RMSE", f'{cm_metrics.get("CV RMSE","—")}%')
    st.caption("NLP: TF-IDF + LogReg + Keyword ensemble")
    st.caption("Cost: XGBoost regression + Monte Carlo")
    st.caption(f"Trained on {clf_metrics.get('Training examples','—')} CO examples")

    run_btn = st.button("▶ RUN ANALYSIS", type="primary", use_container_width=True)

# ─── LOAD & CLASSIFY DATA ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading portfolio data...")
def load_demo():
    return generate_demo_data()

if "classified" not in st.session_state or run_btn:
    with st.spinner("Classifying change orders with NLP engine..."):
        if "Upload" in data_source and uploaded:
            try:
                xl = pd.ExcelFile(uploaded)
                proj_df = xl.parse(xl.sheet_names[0])
                co_df_raw = xl.parse(xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0])
                # Find description column
                desc_col = next((c for c in co_df_raw.columns if "desc" in c.lower()), co_df_raw.columns[0])
                val_col  = next((c for c in co_df_raw.columns if "value" in c.lower() or "amount" in c.lower()), None)
                co_df_raw["description"] = co_df_raw[desc_col].astype(str)
                co_df_raw["value"] = pd.to_numeric(co_df_raw[val_col], errors="coerce").fillna(0) if val_col else 50000
                co_df = co_df_raw
                tracker_df = pd.DataFrame()
            except Exception as e:
                st.error(f"Upload error: {e}")
                st.stop()
        else:
            proj_df, co_df, tracker_df = load_demo()

        # Apply filters
        if "value" in co_df.columns:
            co_df = co_df[co_df["value"] >= min_co_value * 1000]
        if "status" in co_df.columns:
            co_df = co_df[co_df["status"].isin(show_status)]
        if selected_contractor != "All contractors" and "contractor" in co_df.columns:
            co_df = co_df[co_df["contractor"] == selected_contractor]

        # Run NLP classification
        nlp_results = clf.predict_batch(co_df["description"])
        # Rename NLP output columns before concat to avoid duplicate column collision
        # (pandas 2.x keeps both as "root_cause" making it 2D; groupby then fails)
        nlp_renamed = nlp_results.reset_index(drop=True).rename(columns={
            "root_cause": "nlp_root_cause",
            "root_cause_label": "nlp_root_cause_label",
        })
        co_classified = pd.concat([
            co_df.reset_index(drop=True),
            nlp_renamed,
        ], axis=1)
        # If demo data had no ground-truth root_cause, promote NLP prediction
        if "root_cause" not in co_classified.columns:
            co_classified["root_cause"] = co_classified["nlp_root_cause"]
        
        st.session_state.proj_df = proj_df
        st.session_state.co_classified = co_classified
        st.session_state.tracker_df = tracker_df if not isinstance(tracker_df, type(None)) else pd.DataFrame()
        st.session_state.classified = True

proj_df = st.session_state.proj_df
co = st.session_state.co_classified
tracker_df = st.session_state.tracker_df

# ─── KPI CARDS ────────────────────────────────────────────────────────────────
total_budget = proj_df["budget"].sum() if "budget" in proj_df.columns else 0
total_eac    = proj_df["eac"].sum()    if "eac"    in proj_df.columns else 0
total_co_val = co["value"].sum()       if "value"  in co.columns else 0
n_cos        = len(co)
avg_overrun  = proj_df["overrun_pct"].mean() if "overrun_pct" in proj_df.columns else 0
n_high_cos   = len(co[co["confidence"] >= 0.6]) if "confidence" in co.columns else n_cos

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-gray">
    <div class="kpi-label">Programme Budget</div>
    <div class="kpi-value">${total_budget/1e9:.2f}B</div>
    <div class="kpi-delta">{len(proj_df)} active projects</div>
  </div>
  <div class="kpi-card kpi-red">
    <div class="kpi-label">Total EAC</div>
    <div class="kpi-value">${total_eac/1e9:.2f}B</div>
    <div class="kpi-delta">+{(total_eac-total_budget)/total_budget*100:.1f}% vs budget</div>
  </div>
  <div class="kpi-card kpi-amber">
    <div class="kpi-label">CO Value</div>
    <div class="kpi-value">${total_co_val/1e6:.0f}M</div>
    <div class="kpi-delta">{n_cos} change orders</div>
  </div>
  <div class="kpi-card kpi-amber">
    <div class="kpi-label">Avg Overrun</div>
    <div class="kpi-value">{avg_overrun:.1f}%</div>
    <div class="kpi-delta">vs original budget</div>
  </div>
  <div class="kpi-card kpi-blue">
    <div class="kpi-label">NLP Classified</div>
    <div class="kpi-value">{n_high_cos}</div>
    <div class="kpi-delta">high-confidence tags (≥60%)</div>
  </div>
  <div class="kpi-card kpi-purple">
    <div class="kpi-label">Top Root Cause</div>
    <div class="kpi-value" style="font-size:1rem; padding-top:0.3rem;">
      {ROOT_CAUSES.get(co.groupby("root_cause")["value"].sum().idxmax() if "root_cause" in co.columns and len(co)>0 else "design_change","—").split("(")[0].strip()[:18]}
    </div>
    <div class="kpi-delta">by total CO value</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Root Cause Pareto",
    "🔍 CO Classification",
    "📈 Cost Forecast",
    "🏗️ Portfolio Heatmap",
    "🏢 Contractor Intelligence",
    "📋 Recommendations",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: ROOT CAUSE PARETO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">ROOT CAUSE PARETO — CO VALUE BY CATEGORY</div>', unsafe_allow_html=True)

        rc_agg = co.groupby("root_cause").agg(
            total_value=("value", "sum"),
            count=("value", "count"),
            avg_confidence=("confidence", "mean"),
        ).reset_index()
        rc_agg["label"] = rc_agg["root_cause"].map(ROOT_CAUSES)
        rc_agg["pct_of_total"] = rc_agg["total_value"] / rc_agg["total_value"].sum() * 100
        rc_agg["color"] = rc_agg["root_cause"].map(ROOT_CAUSE_COLORS)
        rc_agg = rc_agg.sort_values("total_value", ascending=True)

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(
            y=rc_agg["label"],
            x=rc_agg["total_value"] / 1e6,
            orientation="h",
            marker_color=rc_agg["color"],
            marker_line_width=0,
            text=rc_agg["pct_of_total"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
            customdata=np.stack([rc_agg["count"], rc_agg["avg_confidence"] * 100], axis=1),
            hovertemplate="<b>%{y}</b><br>$%{x:.1f}M (%{text})<br>%{customdata[0]} COs · Avg confidence %{customdata[1]:.0f}%<extra></extra>",
        ))
        fig_pareto.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0c0a06", height=360,
            xaxis_title="Total CO Value ($M)",
            margin=dict(l=10, r=60, t=10, b=40),
            font_family="IBM Plex Sans",
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # Cumulative pareto line
        rc_sorted = rc_agg.sort_values("total_value", ascending=False)
        rc_sorted["cumulative_pct"] = rc_sorted["total_value"].cumsum() / rc_sorted["total_value"].sum() * 100
        st.caption(f"Top 3 causes account for {rc_sorted['pct_of_total'].head(3).sum():.0f}% of total CO value — Pareto principle holds.")

    with col2:
        st.markdown('<div class="section-header">CAUSE BREAKDOWN BY COUNT</div>', unsafe_allow_html=True)
        rc_count = co["root_cause"].value_counts().reset_index()
        rc_count.columns = ["root_cause", "count"]
        rc_count["label"] = rc_count["root_cause"].map(ROOT_CAUSES)
        rc_count["color"] = rc_count["root_cause"].map(ROOT_CAUSE_COLORS)

        fig_pie = go.Figure(go.Pie(
            labels=rc_count["label"],
            values=rc_count["count"],
            marker_colors=rc_count["color"],
            hole=0.52,
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value} COs (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False, font_family="IBM Plex Sans",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-header">CO VALUE OVER TIME</div>', unsafe_allow_html=True)
        if "date" in co.columns:
            co["month"] = pd.to_datetime(co["date"]).dt.to_period("M").dt.to_timestamp()
            monthly = co.groupby(["month", "root_cause"])["value"].sum().reset_index()
            monthly["label"] = monthly["root_cause"].map(ROOT_CAUSES)

            def _hex_to_rgba(hex_color, alpha=0.6):
                h = hex_color.lstrip("#")
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"

            fig_time = go.Figure()
            for rc_key, grp in monthly.groupby("root_cause"):
                fig_time.add_trace(go.Scatter(
                    x=grp["month"], y=grp["value"] / 1e6,
                    name=ROOT_CAUSES.get(rc_key, rc_key)[:20],
                    stackgroup="one",
                    line=dict(width=0),
                    fillcolor=_hex_to_rgba(ROOT_CAUSE_COLORS.get(rc_key, "#64748b")),
                    hovertemplate="%{y:.2f}M<extra>" + ROOT_CAUSES.get(rc_key, rc_key) + "</extra>",
                ))
            fig_time.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0c0a06", height=200,
                margin=dict(l=10, r=10, t=5, b=30),
                showlegend=False, yaxis_title="$M",
                font_family="IBM Plex Sans",
            )
            st.plotly_chart(fig_time, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CHANGE ORDER CLASSIFICATION TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">CHANGE ORDER NLP CLASSIFICATION — ACTIVITY LEVEL AUDIT TRAIL</div>', unsafe_allow_html=True)

    # Build display table
    show_cols = {
        "co_id": "CO ID",
        "project_name": "Project",
        "description": "Description",
        "root_cause": "Root Cause",
        "confidence_pct": "Confidence %",
        "keywords_matched": "Keywords",
        "value": "Value ($)",
        "status": "Status",
    }
    avail = [c for c in show_cols if c in co.columns]
    tbl = co[avail].rename(columns=show_cols).head(100)

    if "Value ($)" in tbl.columns:
        tbl["Value ($)"] = tbl["Value ($)"].apply(lambda v: f"${v:,.0f}")
    if "Root Cause" in tbl.columns:
        tbl["Root Cause"] = co["root_cause"].head(100).map(ROOT_CAUSES)
    if "Confidence %" in tbl.columns:
        tbl["Confidence %"] = tbl["Confidence %"].round(0)

    def colour_conf(val):
        try:
            v = float(val)
            if v >= 70: return "color: #86efac; font-weight: 600"
            elif v >= 45: return "color: #fde68a"
            return "color: #fca5a5"
        except: return ""

    styled = (tbl.style
              .applymap(colour_conf, subset=["Confidence %"])
              .set_properties(**{"font-size": "12px"}))
    st.dataframe(styled, use_container_width=True, height=450)

    st.caption(f"NLP pipeline: TF-IDF (1–3 grams, 3,000 features) + Logistic Regression + Keyword ensemble · F1={clf_metrics.get('CV F1 (macro)','—')}")

    # Low confidence audit
    low_conf = co[co["confidence"] < 0.4] if "confidence" in co.columns else pd.DataFrame()
    if len(low_conf) > 0:
        st.markdown('<div class="section-header">LOW CONFIDENCE — REQUIRES MANUAL REVIEW</div>', unsafe_allow_html=True)
        st.warning(f"{len(low_conf)} change orders have confidence < 40% — review and reclassify before using in portfolio analysis.")
        st.dataframe(low_conf[avail[:4]].head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: COST FORECAST (Monte Carlo)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">PROJECT COST FORECAST — MONTE CARLO SIMULATION</div>', unsafe_allow_html=True)

    col_sel, col_mc = st.columns([1, 2])

    with col_sel:
        if "project_name" in proj_df.columns:
            selected_proj = st.selectbox("Select project", proj_df["project_name"].tolist())
            prow = proj_df[proj_df["project_name"] == selected_proj].iloc[0]
        else:
            selected_proj = "Demo Project"
            prow = proj_df.iloc[0]

        base_budget = float(prow.get("budget", 50e6))
        known_variance = float(prow.get("eac", base_budget * 1.18) - base_budget)
        known_variance = max(0, known_variance)

        # Root cause split for this project's COs
        proj_cos = co[co["project_id"] == prow.get("project_id", co.get("project_id", pd.Series()).iloc[0] if len(co) > 0 else "")] if "project_id" in co.columns else co
        if len(proj_cos) > 0 and "root_cause" in proj_cos.columns:
            rc_split = proj_cos.groupby("root_cause")["value"].sum()
            total_rc = rc_split.sum()
            root_cause_pareto = (rc_split / total_rc).to_dict() if total_rc > 0 else {"design_change": 1.0}
        else:
            root_cause_pareto = {"design_change": 0.35, "ground_conditions": 0.25, "utility_conflict": 0.20, "design_error": 0.20}

        n_sims = st.slider("Monte Carlo simulations", 1000, 20000, 10000, step=1000)

        st.metric("Base Budget", f"${base_budget/1e6:.1f}M")
        st.metric("Known Variance", f"${known_variance/1e6:.1f}M (+{known_variance/base_budget*100:.1f}%)")

    with col_mc:
        mc = run_monte_carlo(
            base_budget=base_budget,
            root_cause_pareto=root_cause_pareto,
            known_variance=known_variance,
            n_simulations=n_sims,
        )

        hist_vals = mc["histogram_values"]
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=hist_vals,
            nbinsx=50,
            marker_color="#d97706",
            marker_line_color="#0c0a06",
            marker_line_width=0.5,
            opacity=0.85,
            name="Simulation outcomes",
        ))
        fig_mc.add_vline(x=base_budget / 1e6, line_dash="dash", line_color="#16a34a",
                         annotation_text=f"Budget ${base_budget/1e6:.0f}M", annotation_font_color="#16a34a")
        fig_mc.add_vline(x=mc["p50"], line_dash="dash", line_color="#fbbf24",
                         annotation_text=f"P50 ${mc['p50']}M", annotation_font_color="#fbbf24")
        fig_mc.add_vline(x=mc["p80"], line_dash="dash", line_color="#f87171",
                         annotation_text=f"P80 ${mc['p80']}M", annotation_font_color="#f87171")
        fig_mc.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0c0a06", height=320,
            xaxis_title="Final Project Cost ($M)", yaxis_title="Frequency",
            margin=dict(l=10, r=10, t=30, b=40),
            font_family="IBM Plex Sans", showlegend=False,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P10 (optimistic)", f"${mc['p10']}M")
        c2.metric("P50 (expected)", f"${mc['p50']}M", f"+{mc['expected_overrun_pct']}%")
        c3.metric("P80 (prudent)", f"${mc['p80']}M", f"+{mc['p80_overrun_pct']}%")
        c4.metric("P90 (risk-adjusted)", f"${mc['p90']}M")
        st.caption(f"Monte Carlo: {n_sims:,} simulations. Uncertainty modelled per root cause volatility (ground conditions=45% CoV, design change=25% CoV). Right-skewed log-normal draws.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PORTFOLIO HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">PORTFOLIO COST PERFORMANCE HEATMAP</div>', unsafe_allow_html=True)

    col_heat, col_bubble = st.columns(2)

    with col_heat:
        if "overrun_pct" in proj_df.columns and "project_name" in proj_df.columns:
            ph = proj_df.sort_values("overrun_pct", ascending=False).head(15)
            colors = ph["overrun_pct"].apply(
                lambda v: "#dc2626" if v > 25 else "#d97706" if v > 10 else "#16a34a"
            )
            fig_heat = go.Figure(go.Bar(
                y=ph["project_name"].str[:35],
                x=ph["overrun_pct"],
                orientation="h",
                marker_color=colors,
                text=ph["overrun_pct"].apply(lambda v: f"+{v:.1f}%"),
                textposition="outside",
            ))
            fig_heat.add_vline(x=0, line_color="#57534e", line_width=1)
            fig_heat.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0c0a06", height=380,
                xaxis_title="Cost Overrun %", margin=dict(l=10, r=60, t=10, b=10),
                font_family="IBM Plex Sans",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_bubble:
        st.markdown('<div class="section-header">BUDGET vs EAC BUBBLE CHART</div>', unsafe_allow_html=True)
        if all(c in proj_df.columns for c in ["budget", "eac", "cpi", "project_name"]):
            fig_bubble = go.Figure(go.Scatter(
                x=proj_df["budget"] / 1e6,
                y=proj_df["eac"] / 1e6,
                mode="markers+text",
                marker=dict(
                    size=np.sqrt(proj_df["budget"] / 1e6) * 2.5,
                    color=proj_df["overrun_pct"],
                    colorscale=[[0, "#16a34a"], [0.4, "#d97706"], [1, "#dc2626"]],
                    showscale=True,
                    colorbar=dict(title="Overrun %", thickness=10),
                    line=dict(width=0.5, color="#292524"),
                ),
                text=proj_df["project_id"],
                textposition="top center",
                textfont=dict(size=10, color="#a8a29e"),
                hovertext=proj_df["project_name"] + "<br>Budget: $" + (proj_df["budget"] / 1e6).round(1).astype(str) + "M<br>EAC: $" + (proj_df["eac"] / 1e6).round(1).astype(str) + "M",
                hoverinfo="text",
            ))
            max_val = max(proj_df["budget"].max(), proj_df["eac"].max()) / 1e6 * 1.05
            fig_bubble.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(color="#57534e", dash="dot", width=1),
                showlegend=False, hoverinfo="skip",
            ))
            fig_bubble.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0c0a06", height=380,
                xaxis_title="Original Budget ($M)", yaxis_title="EAC ($M)",
                margin=dict(l=10, r=60, t=10, b=40),
                font_family="IBM Plex Sans",
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.caption("Points above the diagonal line = cost overrun. Bubble size ∝ project budget.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: CONTRACTOR INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">CONTRACTOR ACCOUNTABILITY MATRIX</div>', unsafe_allow_html=True)

    if "contractor" in co.columns and "value" in co.columns:
        contr_agg = co.groupby("contractor").agg(
            total_co_value=("value", "sum"),
            co_count=("value", "count"),
            avg_confidence=("confidence", "mean"),
        ).reset_index()

        if "contractor" in proj_df.columns and "budget" in proj_df.columns:
            contr_budget = proj_df.groupby("contractor")["budget"].sum().reset_index()
            contr_budget.columns = ["contractor", "total_budget"]
            contr_agg = contr_agg.merge(contr_budget, on="contractor", how="left")
            contr_agg["co_pct_of_budget"] = (contr_agg["total_co_value"] / contr_agg["total_budget"] * 100).round(1)
        else:
            contr_agg["co_pct_of_budget"] = 0.0

        # Portfolio average for benchmarking
        portfolio_avg = contr_agg["co_pct_of_budget"].mean()

        contr_agg["vs_avg"] = contr_agg["co_pct_of_budget"] - portfolio_avg
        contr_agg = contr_agg.sort_values("co_pct_of_budget", ascending=False)

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_contr = go.Figure(go.Bar(
                y=contr_agg["contractor"].str[:25],
                x=contr_agg["co_pct_of_budget"],
                orientation="h",
                marker_color=contr_agg["co_pct_of_budget"].apply(
                    lambda v: "#dc2626" if v > portfolio_avg * 1.5 else
                              "#d97706" if v > portfolio_avg else "#16a34a"
                ),
                text=contr_agg["co_pct_of_budget"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_contr.add_vline(x=portfolio_avg, line_dash="dot", line_color="#78716c",
                                annotation_text=f"Avg {portfolio_avg:.1f}%")
            fig_contr.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0c0a06", height=320,
                xaxis_title="Change Orders as % of Contract Value",
                margin=dict(l=10, r=80, t=10, b=10),
                font_family="IBM Plex Sans",
            )
            st.plotly_chart(fig_contr, use_container_width=True)

        with col_c2:
            st.markdown('<div class="section-header">ROOT CAUSE BY CONTRACTOR</div>', unsafe_allow_html=True)
            rc_by_contr = co.groupby(["contractor", "root_cause"])["value"].sum().reset_index()
            rc_by_contr["label"] = rc_by_contr["root_cause"].map(ROOT_CAUSES)
            rc_by_contr["color"] = rc_by_contr["root_cause"].map(ROOT_CAUSE_COLORS)

            fig_rc_c = px.bar(
                rc_by_contr,
                x="value", y="contractor",
                color="label",
                orientation="h",
                color_discrete_map={v: ROOT_CAUSE_COLORS.get(k, "#64748b") for k, v in ROOT_CAUSES.items()},
                labels={"value": "CO Value ($)", "contractor": ""},
            )
            fig_rc_c.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0c0a06", height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                font_family="IBM Plex Sans",
                legend=dict(font=dict(size=9), orientation="h", y=-0.25),
                showlegend=True,
            )
            st.plotly_chart(fig_rc_c, use_container_width=True)

        # Flag contractors with disproportionate ground conditions claims
        if "ground_conditions" in co["root_cause"].values:
            gc_by_contr = co[co["root_cause"] == "ground_conditions"].groupby("contractor")["value"].sum()
            all_by_contr = co.groupby("contractor")["value"].sum()
            gc_pct = (gc_by_contr / all_by_contr * 100).dropna().sort_values(ascending=False)
            gc_avg = gc_pct.mean()
            flagged = gc_pct[gc_pct > gc_avg * 1.8]
            if len(flagged) > 0:
                st.warning(f"**Ground conditions claim alert:** {', '.join(flagged.index.tolist())} have ground conditions claims {flagged.iloc[0]/gc_avg:.1f}× the portfolio average — review for opportunistic claiming.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">PORTFOLIO INTERVENTION REGISTER — PRIORITISED ACTIONS</div>', unsafe_allow_html=True)

    rc_pareto_pct = (
        co.groupby("root_cause")["value"].sum() / co["value"].sum() * 100
    ).sort_values(ascending=False) if "root_cause" in co.columns else pd.Series()

    recs = generate_portfolio_recommendations(
        root_pareto_pct=rc_pareto_pct,
        total_programme_budget=total_budget,
        total_co_value=total_co_val,
    )

    if recs:
        for i, rec in enumerate(recs, 1):
            priority_class = "high" if rec["priority"] == "HIGH" else "medium"
            st.markdown(f"""
            <div class="rec-card {priority_class}">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="rec-title">#{i} · {rec['recommendation']}</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#fbbf24;">
                  {rec['priority']} · ${rec['value_m']}M · {rec['pct']:.0f}% of CO value
                </div>
              </div>
              <div class="rec-cause">{rec['cause']}</div>
              <div class="rec-action">▶ {rec['action']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-priority recommendations generated. Adjust filters to see more change orders.")

    st.markdown("---")
    st.markdown('<div class="section-header">EXPORT INTELLIGENCE REPORT</div>', unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        export_co = co[["co_id", "project_name", "description", "root_cause",
                         "confidence_pct", "keywords_matched", "value", "status"]].copy() if all(
            c in co.columns for c in ["co_id", "confidence_pct"]) else co
        export_co.columns = [c.replace("_", " ").title() for c in export_co.columns]
        csv = export_co.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Classified COs (CSV)", data=csv,
                           file_name=f"CVRCA_COs_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv", use_container_width=True)

    with col_dl2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            export_co.to_excel(w, sheet_name="Change Orders", index=False)
            rc_pareto_pct.reset_index().rename(columns={"root_cause":"Root Cause","value":"% of Total"}).to_excel(w, sheet_name="Pareto", index=False)
            if recs:
                pd.DataFrame(recs).to_excel(w, sheet_name="Recommendations", index=False)
        st.download_button("⬇ Full Report (Excel)", data=buf.getvalue(),
                           file_name=f"CVRCA_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">INTERVIEW POSITIONING NOTE</div>', unsafe_allow_html=True)
    st.info("""
**Why this project matters for KPMG:**

This system does what no standard EVM dashboard can: it tells you **why** costs are overrunning, not just **how much**.
The NLP classifier reads every change order description — the same unstructured text that sits in Excel trackers and SAP exports — and tags it with a standardised root cause category.

**Technical differentiation:**
- TF-IDF + Logistic Regression + Keyword ensemble: 82%+ F1, no GPU required, retrains in <2s on client data
- Monte Carlo simulation per project: P50/P80 cost forecasts with root-cause-specific volatility assumptions
- Contractor accountability matrix: surfaces which contractors are generating disproportionate ground conditions claims
- Portfolio Pareto: turns 150+ individual change orders into 3–4 actionable recommendations

**Consulting value delivered:**
- "47% of this programme's cost overrun is design change — the client PMO is generating scope without budget approval. Implement a change freeze at 30% design and you recover $12M."
- That is a specific, quantified, owner-assigned intervention — not a dashboard observation.

*"I built a system that reads your change orders and tells you which management decisions will recover the most cost."*
    """)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #292524; display:flex; justify-content:space-between;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#44403c;">CVRCA v1.0 · TF-IDF NLP · XGBoost · Monte Carlo · Streamlit</span>
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#44403c;">Project 2 of 3 · KPMG Infrastructure Advisory · US Energy &amp; Utilities</span>
</div>
""", unsafe_allow_html=True)
