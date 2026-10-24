"""
GlucoTwin - doctor dashboard for a Type 2 diabetes digital twin.

Run:  streamlit run app.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import LABELS, SPIKE_MGDL, build_features  # noqa: E402
from physiology import (MEALS, STEP_MIN, STEPS_PER_DAY, patient_params,  # noqa: E402
                        simulate)

ROOT = os.path.dirname(__file__)
INK, TEAL, AMBER, RED, MIST = "#16323A", "#2A7F7A", "#D9962B", "#B8433A", "#E6EEEF"

st.set_page_config(page_title="GlucoTwin", page_icon="🩺", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Source Sans 3', system-ui, sans-serif; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.risk {{ border-left: 6px solid var(--c); padding: 0.6rem 1rem; background: {MIST}; border-radius: 4px; }}
.risk b {{ font-size: 2.2rem; color: var(--c); }}
.note {{ color: #5A6B70; font-size: 0.9rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    d = os.path.join(ROOT, "data")
    return (pd.read_csv(os.path.join(d, "ehr.csv")),
            pd.read_csv(os.path.join(d, "wearables.csv"), parse_dates=["timestamp"]),
            pd.read_csv(os.path.join(d, "meals.csv")))


@st.cache_resource
def load_model():
    bundle = joblib.load(os.path.join(ROOT, "models", "glucotwin.joblib"))
    with open(os.path.join(ROOT, "models", "metrics.json")) as fh:
        bundle["metrics"] = json.load(fh)
    return bundle


@st.cache_data
def patient_features(pid):
    ehr, wear, meals = load()
    row = ehr[ehr.patient_id == pid].iloc[0].to_dict()
    return build_features(wear[wear.patient_id == pid], row,
                          meals[meals.patient_id == pid], with_label=False)


def explain(model, x, medians, features):
    """Which inputs push this patient's risk up? (swap each for a typical value)"""
    base = model.predict_proba(x[features])[0, 1]
    groups = {}
    for f in features:
        groups.setdefault(LABELS[f], []).append(f)
    rows = []
    for label, cols in groups.items():
        x2 = x[features].copy()
        for c in cols:
            x2[c] = medians[c]
        rows.append((label, base - model.predict_proba(x2)[0, 1]))
    return pd.DataFrame(rows, columns=["driver", "effect"]).sort_values("effect", ascending=False)


if not os.path.exists(os.path.join(ROOT, "models", "glucotwin.joblib")):
    st.error("No trained model found. Run `python src/generate_data.py` then `python src/train.py`.")
    st.stop()

ehr, wear, meals = load()
bundle = load_model()
model, FEATS = bundle["model"], bundle["features"]
test_ids = set(bundle["test_patients"])

# ---------- Sidebar: choose patient and moment ----------
with st.sidebar:
    st.header("Patient")
    options = ehr.patient_id.tolist()
    pid = st.selectbox(
        "Virtual patient", options,
        format_func=lambda p: f"{p} · {ehr.set_index('patient_id').loc[p, 'name']}"
        + ("  (unseen by model)" if p in test_ids else ""))
    day = st.slider("Day of monitoring", 2, 14, 7)
    hour = st.slider("Time now (hour)", 0.0, 23.5, 18.5, step=0.5, format="%.1f h")
    st.caption("Patients marked 'unseen by model' were held out of training.")

e = ehr.set_index("patient_id").loc[pid]
feats = patient_features(pid)
now = (day - 1) * STEPS_PER_DAY + int(hour * 60 / STEP_MIN)
x_now = feats[feats.step == now]
risk = float(model.predict_proba(x_now[FEATS])[0, 1])
g_now = float(x_now["glucose"].iloc[0])

# ---------- Header ----------
st.title(f"{e['name']}")
st.markdown(f"<span class='note'>{e['age']} y · {e['sex']} · BMI {e['bmi']} · "
            f"HbA1c {e['hba1c']}% · diagnosed {e['years_since_diagnosis']} y ago · "
            f"{'on metformin' if e['on_metformin'] else 'no metformin'}</span>",
            unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.2, 1, 1])
if g_now >= 160:
    colour, verdict = AMBER, "Already elevated — forecast applies below 160 mg/dL"
elif risk >= bundle["threshold"]:
    colour, verdict = RED, "Spike likely within 2 hours"
elif risk >= 0.25:
    colour, verdict = AMBER, "Watch closely"
else:
    colour, verdict = TEAL, "Stable for the next 2 hours"
c1.markdown(f"<div class='risk' style='--c:{colour}'>Spike risk (next 2 h)<br>"
            f"<b>{risk:.0%}</b><br>{verdict}</div>", unsafe_allow_html=True)
c2.metric("Glucose now", f"{g_now:.0f} mg/dL",
          f"{x_now['slope_30'].iloc[0] * 30:+.0f} in 30 min", delta_color="inverse")
c3.metric("Last night's sleep", f"{x_now['last_night_sleep_h'].iloc[0]:.1f} h",
          f"{x_now['steps_120'].iloc[0]:.0f} steps in 2 h", delta_color="off")

# ---------- Last 24 hours ----------
st.subheader("Last 24 hours")
w = wear[(wear.patient_id == pid) & (wear.step > now - STEPS_PER_DAY) & (wear.step <= now)]
m = meals[(meals.patient_id == pid) & (meals.step > now - STEPS_PER_DAY) & (meals.step <= now)]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=w.timestamp, y=w.glucose_mgdl, name="Glucose", line=dict(color=INK, width=2)), 1, 1)
fig.add_hline(y=SPIKE_MGDL, line=dict(color=RED, dash="dot"), row=1, col=1)
mt = w.set_index("step").reindex(m.step)
fig.add_trace(go.Scatter(x=mt.timestamp, y=mt.glucose_mgdl, mode="markers", name="Meal",
                         marker=dict(color=AMBER, size=10, symbol="triangle-up"),
                         text=m.meal.str.replace("_", " "), hovertemplate="%{text}<extra></extra>"), 1, 1)
fig.add_trace(go.Scatter(x=w.timestamp, y=w.heart_rate, name="Heart rate", line=dict(color=TEAL, width=1)), 2, 1)
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"),
                  plot_bgcolor="white")
fig.update_yaxes(title_text="mg/dL", row=1, col=1)
fig.update_yaxes(title_text="bpm", row=2, col=1)
st.plotly_chart(fig, width="stretch")

# ---------- Why ----------
left, right = st.columns([1.3, 1])
with left:
    st.subheader("What is driving this risk")
    ex = explain(model, x_now, bundle["medians"], FEATS).head(5)
    bar = go.Figure(go.Bar(x=ex.effect * 100, y=ex.driver, orientation="h",
                           marker_color=[RED if v > 0 else TEAL for v in ex.effect]))
    bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      xaxis_title="Change in risk vs a typical patient (percentage points)",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, width="stretch")
with right:
    st.subheader("Health record")
    st.table(pd.DataFrame({
        "Item": ["Fasting glucose", "LDL", "Systolic BP", "Hypertension", "Family history of T2D",
                 "TCF7L2 risk allele", "Diet pattern"],
        "Value": [f"{e['fasting_glucose']} mg/dL", f"{e['ldl']} mg/dL", f"{e['systolic_bp']} mmHg",
                  "Yes" if e["hypertension_dx"] else "No", "Yes" if e["family_history_t2d"] else "No",
                  "Present" if e["tcf7l2_risk_allele"] else "Absent",
                  e["diet_pattern"].replace("_", " ")]}).set_index("Item"))

# ---------- What-if twin ----------
st.subheader("Try a change on the virtual patient")
st.markdown("<span class='note'>Simulates the next 6 hours from the current reading. "
            "Compare the patient's usual next meal with an alternative plan.</span>",
            unsafe_allow_html=True)

params = patient_params(e.to_dict())
usual = meals[meals.patient_id == pid].copy()
usual["sod"] = usual.step % STEPS_PER_DAY
hist_now = now % STEPS_PER_DAY
upcoming = usual[(usual.sod > hist_now) & (usual.sod <= hist_now + 72)]
usual_meal = upcoming.meal.mode().iloc[0] if len(upcoming) else "rice_sambar_rasam"
usual_in = int(((upcoming.sod.median() - hist_now) * STEP_MIN)) if len(upcoming) else 60

a, b, c, d = st.columns(4)
meal_names = list(MEALS.keys())
alt_meal = a.selectbox("Next meal", meal_names, index=meal_names.index("millet_meal"),
                       format_func=lambda s: s.replace("_", " "))
meal_in = b.slider("Eaten in (min)", 0, 240, min(max(usual_in, 0), 240), step=15)
walk = c.slider("Walk after meal (min)", 0, 45, 20, step=5)
sleep = d.slider("Sleep tonight-equivalent (h)", 4.0, 9.0,
                 float(x_now["last_night_sleep_h"].iloc[0]), step=0.5)

H = 72
def run(meal, walk_min, sleep_h):
    idx = meal_in // STEP_MIN
    steps = np.full(H, 10.0)
    if walk_min:
        steps[idx + 4: idx + 4 + walk_min // STEP_MIN] += 450
    return simulate(params, H, {idx: MEALS[meal]}, steps, sleep_debt_hours=7.5 - sleep_h,
                    start_glucose=g_now, start_step_of_day=hist_now, noise=False)

base_curve = run(usual_meal, 0, float(x_now["last_night_sleep_h"].iloc[0]))
alt_curve = run(alt_meal, walk, sleep)
t = pd.Timestamp(x_now["timestamp"].iloc[0]) + pd.to_timedelta(np.arange(H) * STEP_MIN, "min")

wf = go.Figure()
wf.add_trace(go.Scatter(x=t, y=base_curve, name=f"Usual: {usual_meal.replace('_', ' ')}, no walk",
                        line=dict(color=INK, dash="dash")))
wf.add_trace(go.Scatter(x=t, y=alt_curve, name=f"Plan: {alt_meal.replace('_', ' ')} + {walk} min walk",
                        line=dict(color=TEAL, width=3)))
wf.add_hline(y=SPIKE_MGDL, line=dict(color=RED, dash="dot"))
wf.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                 legend=dict(orientation="h"), yaxis_title="mg/dL")
st.plotly_chart(wf, width="stretch")

def tir(curve):
    return (curve > SPIKE_MGDL).sum() * STEP_MIN
k1, k2, k3 = st.columns(3)
k1.metric("Peak glucose", f"{alt_curve.max():.0f} mg/dL", f"{alt_curve.max() - base_curve.max():+.0f} vs usual",
          delta_color="inverse")
k2.metric("Minutes above 180", f"{tir(alt_curve)}", f"{tir(alt_curve) - tir(base_curve):+d} vs usual",
          delta_color="inverse")
k3.metric("Model accuracy (held-out patients)", f"AUROC {bundle['metrics']['auroc']}",
          f"{bundle['metrics']['spikes_warned_30min_ahead']:.0%} of spikes flagged ≥30 min early",
          delta_color="off")

st.caption("Proof of concept on synthetic data. Not for clinical use.")
