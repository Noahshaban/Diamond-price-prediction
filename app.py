"""
app.py  —  Diamond Price Predictor · Streamlit
Layout: experiment dashboard (top) → live predictor (bottom)
Run:    streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Diamond Price Predictor",
    
    layout="wide",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e1e;
    border: 1px solid #0056b3;
    border-radius: 10px;
    padding: 16px 20px;
    transition: border-color 0.3s ease;
}
.metric-card:hover {
    border-color: #00bfff;
}
.metric-label { font-size: 13px; color: #ffffff; margin: 0 0 4px; }
.metric-value { font-size: 26px; font-weight: 600; margin: 0; color: #ffffff; }
.metric-sub   { font-size: 11px; color: #ffffff; margin: 4px 0 0; }

.bar-row   { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.bar-label { font-size:13px; color:#ffffff; width:140px; flex-shrink:0; }
.bar-bg    { background:#2a2a2a; border-radius:99px; height:8px; flex:1; overflow:hidden; }
.bar-fill  { height:100%; border-radius:99px; }
.bar-pct   { font-size:13px; font-weight:500; min-width:46px; text-align:right; color:#ffffff; }

.param-row      { display:flex; justify-content:space-between; align-items:center;
                  padding:7px 0; border-bottom:1px solid #0056b3; font-size:13px; transition: border-color 0.3s ease; }
.param-row:hover { border-bottom-color: #00bfff; }
.param-row:last-child { border-bottom:none; }
.param-key { color:#ffffff; }
.param-val { background:#2a2a2a; border-radius:6px; padding:2px 10px;
             color:#ffffff; font-size:12px; }

.step-row  { display:flex; align-items:center; gap:12px;
             padding:7px 0; font-size:13px; color:#ffffff; }
.step-num  { width:24px; height:24px; border-radius:50%; background:#2a2a2a;
             display:flex; align-items:center; justify-content:center;
             font-size:11px; font-weight:600; color:#ffffff; flex-shrink:0; }
.step-num.best { background:#1a3a2a; color:#4caf7d; }
.step-name { font-weight:500; color:#ffffff; flex:1; }
.step-note { font-size:11px; color:#ffffff; }

.tag          { display:inline-block; background:#2a2a2a; border-radius:6px;
                padding:4px 10px; font-size:12px; color:#ffffff; margin:3px 3px 3px 0; 
                border: 1px solid transparent; transition: border-color 0.3s ease; }
.tag:hover    { border-color: #00bfff; }
.tag.hi       { background:#1a3a2a; color:#4caf7d; }
.tag.removed  { background:#2a1a1a; color:#ff6b6b; text-decoration:line-through; }
.tag.target   { background:#1a2a3a; color:#5b9bd5; }

.section-card  { background:#181818; border:1px solid #0056b3; border-radius:12px;
                 padding:20px 22px; height:100%; transition: border-color 0.3s ease; }
.section-card:hover { border-color: #00bfff; }
.section-title { font-size:14px; font-weight:500; color:#ffffff; margin:0 0 16px; }

.takeaway      { background:#1a2e1f; border:1px solid #0056b3; border-radius:10px;
                 padding:16px 20px; margin-top:16px; transition: border-color 0.3s ease; }
.takeaway:hover { border-color: #00bfff; }
.takeaway-head { font-size:12px; font-weight:600; color:#4caf7d; margin:0 0 6px; }
.takeaway-body { font-size:13px; color:#ffffff; margin:0; }
.badge-green   { background:#1a3a2a; color:#4caf7d; border-radius:99px;
                 padding:4px 12px; font-size:12px; font-weight:500; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXPERIMENT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("###  Diamond price prediction")
st.caption("Kaggle · Diamonds Dataset · 53,940 samples")

# top metric cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><p class="metric-label">Best R² score</p>'
                '<p class="metric-value" style="color:#4caf7d;">98.1%</p>'
                '<p class="metric-sub">XGBoost · test set</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><p class="metric-label">Baseline R²</p>'
                '<p class="metric-value">90.3%</p>'
                '<p class="metric-sub">Linear Regression</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><p class="metric-label">MSE improvement</p>'
                '<p class="metric-value" style="color:#5b9bd5;">5×</p>'
                '<p class="metric-sub">1.5M → 302K</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><p class="metric-label">Training samples</p>'
                '<p class="metric-value">43,152</p>'
                '<p class="metric-sub">80 / 20 split</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# model comparison + params
left, right = st.columns([1.4, 1])

with left:
    st.markdown("""
    <div class="section-card">
      <p class="section-title">Model comparison — R² score</p>
      <div class="bar-row">
        <span class="bar-label">Linear Regression</span>
        <div class="bar-bg"><div class="bar-fill" style="width:90.3%;background:#555;"></div></div>
        <span class="bar-pct" style="color:#aaa;">90.3%</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">SGD Regressor</span>
        <div class="bar-bg"><div class="bar-fill" style="width:90%;background:#444;"></div></div>
        <span class="bar-pct" style="color:#aaa;">~90%</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">XGBoost</span>
        <div class="bar-bg"><div class="bar-fill" style="width:98.1%;background:#4caf7d;"></div></div>
        <span class="bar-pct" style="color:#4caf7d;">98.1%</span>
      </div>
      <div style="margin-top:16px;padding-top:14px;border-top:1px solid #2a2a2a;">
        <p style="font-size:11px;color:#555;margin:0 0 8px;">MSE · lower is better</p>
        <span style="font-size:12px;color:#666;margin-right:16px;">⬤ LR: 1,537,393</span>
        <span style="font-size:12px;color:#4caf7d;">⬤ XGB: 302,921</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="section-card">
      <p class="section-title">XGBoost parameters</p>
      <div class="param-row"><span class="param-key">n_estimators</span><span class="param-val">300</span></div>
      <div class="param-row"><span class="param-key">learning_rate</span><span class="param-val">0.1</span></div>
      <div class="param-row"><span class="param-key">max_depth</span><span class="param-val">6</span></div>
      <div class="param-row"><span class="param-key">subsample</span><span class="param-val">0.8</span></div>
      <div class="param-row"><span class="param-key">colsample_bytree</span><span class="param-val">0.8</span></div>
      <div class="param-row"><span class="param-key">random_state</span><span class="param-val">42</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# pipeline + features
b1, b2 = st.columns(2)

with b1:
    st.markdown("""
    <div class="section-card">
      <p class="section-title">Pipeline steps</p>
      <div class="step-row"><div class="step-num">1</div>
        <span class="step-name">Load data</span><span class="step-note">drop x, y, z</span></div>
      <div class="step-row"><div class="step-num">2</div>
        <span class="step-name">Ordinal encoding</span><span class="step-note">cut · color · clarity</span></div>
      <div class="step-row"><div class="step-num">3</div>
        <span class="step-name">Train / test split</span><span class="step-note">80 / 20 · seed 42</span></div>
      <div class="step-row"><div class="step-num">4</div>
        <span class="step-name">Standard scaling</span><span class="step-note">linear models only</span></div>
      <div class="step-row"><div class="step-num best">5</div>
        <span class="step-name">XGBoost fit</span><span class="step-note">no scaling needed</span></div>
    </div>
    """, unsafe_allow_html=True)

with b2:
    st.markdown("""
    <div class="section-card">
      <p class="section-title">Features used</p>
      <div style="margin-bottom:16px;">
        <span class="tag hi">carat</span>
        <span class="tag">cut</span><span class="tag">color</span>
        <span class="tag">clarity</span><span class="tag">depth</span>
        <span class="tag">table</span>
        <span class="tag removed">x</span>
        <span class="tag removed">y</span>
        <span class="tag removed">z</span>
      </div>
      <div style="border-top:1px solid #2a2a2a;padding-top:12px;">
        <p style="font-size:11px;color:#555;margin:0 0 8px;">Target variable</p>
        <span class="tag target">price (USD)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# takeaway
st.markdown("""
<div class="takeaway">
  <p class="takeaway-head">Key takeaway</p>
  <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
    <p class="takeaway-body">
      Switching from Linear Regression to XGBoost reduced prediction error by 5× —
      from ~$1,240 average error to ~$550 — on the same features and data.
    </p>
    <span class="badge-green">+7.8 pp improvement</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("### Try the model")
st.caption("Adjust the diamond specs and get an instant price prediction.")

@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl"), joblib.load("model/encoder.pkl")

try:
    model, encoder = load_model()
    model_loaded = True
except FileNotFoundError:
    st.warning("Model not found — run `python train.py` first.")
    model_loaded = False

if model_loaded:
    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.number_input("Carat weight", min_value=0.2, max_value=5.0,
                                value=1.0, step=0.01)
        cut   = st.selectbox("Cut quality",
                             ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)

    with col2:
        color   = st.selectbox("Color grade",
                               ["J", "I", "H", "G", "F", "E", "D"], index=6)
        clarity = st.selectbox("Clarity grade",
                               ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=7)

    with col3:
        depth = st.number_input("Depth %", min_value=40.0, max_value=80.0,
                                value=61.5, step=0.1)
        table = st.number_input("Table %", min_value=40.0, max_value=100.0,
                                value=57.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict price  →", use_container_width=True, type="primary"):
        cat = encoder.transform([[cut, color, clarity]])[0]
        features = np.array([[carat, cat[0], cat[1], cat[2], depth, table]])
        price = float(model.predict(features)[0])

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Estimated price", f"${price:,.0f}")
        r2.metric("Low estimate",    f"${price * 0.90:,.0f}")
        r3.metric("High estimate",   f"${price * 1.10:,.0f}")
        r4.metric("Model R²",        "98.1%")

        st.caption("Price range based on ±10% confidence band · avg model error ~$550")

        with st.expander("Input summary"):
            st.dataframe(pd.DataFrame({
                "Feature": ["Carat", "Cut", "Color", "Clarity", "Depth %", "Table %"],
                "Value":   [carat,   cut,   color,   clarity,   depth,     table],
            }), hide_index=True, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Model: XGBoost · Dataset: Kaggle Diamonds (53,940 rows) · Pipeline: OrdinalEncoder → XGBRegressor")
