"""Draw the GlucoTwin architecture diagram to docs/architecture.pdf (and .png)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, TEAL, AMBER, MIST = "#16323A", "#2A7F7A", "#D9962B", "#E6EEEF"
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")


def box(x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=MIST, ec=edge, lw=2))
    ax.text(x + 0.15, y + h - 0.3, title, fontsize=11, weight="bold", color=INK, va="top")
    ax.text(x + 0.15, y + h - 0.7, body, fontsize=8.6, color=INK, va="top", linespacing=1.45)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))


ax.text(0.2, 6.2, "GlucoTwin — Type 2 diabetes digital twin (proof of concept)",
        fontsize=15, weight="bold", color=INK)

box(0.2, 3.4, 3.2, 2.4, "Static stream: EHR",
    "Synthetic patients (Synthea-style)\nAge, sex, BMI, HbA1c\nFasting glucose, LDL, BP\n"
    "Diagnoses, medication\nTCF7L2 genetic marker\nDiet pattern", AMBER)
box(0.2, 0.4, 3.2, 2.6, "Dynamic stream: sensors",
    "5-min time series, 14 days\nCGM glucose\nHeart rate, HRV\nSteps\nSleep stages and duration\n"
    "Meal log (Indian meals)", AMBER)
box(3.9, 1.6, 3.0, 3.0, "Fusion and features",
    "Patient-level EHR features\njoined to every time step\n\nPast-only rolling features:\n"
    "glucose lags, trend, variability\nactivity, HR/HRV, sleep\ncarbs eaten, time to usual meal",
    TEAL)
box(7.4, 3.4, 2.7, 2.4, "Prediction model",
    "Gradient-boosted trees\n(scikit-learn)\n\nOutput: risk that glucose\ncrosses 180 mg/dL\nin the next 2 hours", TEAL)
box(7.4, 0.4, 2.7, 2.6, "Twin simulator",
    "Personal physiology from EHR:\nbaseline, insulin resistance,\nclearance\n\nRuns what-if scenarios:\n"
    "meal swap, walk, sleep", TEAL)
box(10.6, 1.6, 2.3, 3.0, "Doctor dashboard",
    "Streamlit\n\nRisk score + status\nLast 24 h glucose / HR\nRisk drivers\nHealth record\n"
    "What-if comparison", INK)

arrow(3.4, 4.6, 3.9, 3.6); arrow(3.4, 1.7, 3.9, 2.6)
arrow(6.9, 3.6, 7.4, 4.4); arrow(6.9, 2.2, 7.4, 1.8)
arrow(10.1, 4.4, 10.6, 3.6); arrow(10.1, 1.7, 10.6, 2.6)

ax.text(0.2, 0.05, "All data synthetic — no real patient data (DPDP Act / HIPAA compliant sandbox).",
        fontsize=8.5, color="#5A6B70", style="italic")

out = os.path.dirname(__file__)
fig.savefig(os.path.join(out, "architecture.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out, "architecture.png"), dpi=160, bbox_inches="tight")
print("Saved docs/architecture.pdf and docs/architecture.png")
