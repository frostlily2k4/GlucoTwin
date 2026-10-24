"""
Simple glucose physiology model for a Type 2 diabetes digital twin.

This is NOT a clinical model. It is a transparent, explainable approximation
used for two things:
  1. Generating synthetic wearable (CGM) data for training.
  2. Running "what-if" simulations on a virtual patient in the dashboard.

Each patient has personal parameters derived from their EHR:
  - basal glucose (from HbA1c)
  - insulin resistance (from BMI, HbA1c, South Asian phenotype factor)
  - clearance rate (how fast glucose returns to baseline)

Time step = 5 minutes.
"""
import numpy as np

STEP_MIN = 5
STEPS_PER_DAY = 24 * 60 // STEP_MIN  # 288

# Typical Indian meals: (name, carbs in grams, glycemic speed factor)
MEALS = {
    "idli_sambar": (55, 1.0),
    "dosa_chutney": (60, 1.1),
    "pongal": (65, 1.05),
    "rice_sambar_rasam": (95, 1.15),
    "biryani": (110, 1.2),
    "chapati_dal": (60, 0.85),
    "millet_meal": (55, 0.7),
    "curd_rice": (70, 1.0),
    "tea_biscuits": (25, 1.3),
}


def patient_params(ehr: dict) -> dict:
    """Derive physiological parameters from static EHR fields."""
    hba1c = ehr["hba1c"]
    bmi = ehr["bmi"]
    # Estimated average glucose from HbA1c (ADAG formula), used as baseline anchor
    eag = 28.7 * hba1c - 46.7
    basal = 0.62 * eag + 18
    # South Asian phenotype: higher insulin resistance at lower BMI
    ir = 0.6 + 0.08 * (hba1c - 5.5) + 0.025 * max(bmi - 23, 0)
    if ehr.get("family_history_t2d", 0):
        ir += 0.08
    if ehr.get("on_metformin", 0):
        ir *= 0.85
    clearance = 0.035 / (0.7 + 0.3 * ir)  # fraction back toward basal per step
    return {"basal": basal, "ir": ir, "clearance": clearance}


def meal_kernel(length=36):
    """Carb absorption shape over 3 hours (gamma-like, peaks ~45-60 min)."""
    t = np.arange(length) * STEP_MIN
    k = (t / 50.0) * np.exp(1 - t / 50.0)
    return k / k.sum()


KERNEL = meal_kernel()


def simulate(params, n_steps, meals, steps_per_slot, sleep_debt_hours=0.0,
             start_glucose=None, start_step_of_day=0, rng=None, noise=True):
    """
    Simulate glucose.

    meals: dict {step_index: (carbs_g, speed_factor)}
    steps_per_slot: array of step counts per 5-min slot (activity)
    sleep_debt_hours: last night's sleep shortfall vs 7.5h (raises resistance)
    """
    rng = rng or np.random.default_rng()
    basal = params["basal"]
    ir = params["ir"] * (1 + 0.06 * max(sleep_debt_hours, 0))
    clr = params["clearance"]

    carb_input = np.zeros(n_steps + len(KERNEL))
    for idx, (carbs, speed) in meals.items():
        if 0 <= idx < n_steps:
            k = KERNEL ** speed
            k = k / k.sum()
            carb_input[idx:idx + len(k)] += carbs * k

    g = np.zeros(n_steps)
    cur = start_glucose if start_glucose is not None else basal
    for i in range(n_steps):
        hour = ((start_step_of_day + i) % STEPS_PER_DAY) * STEP_MIN / 60
        dawn = 0.25 if 4 <= hour < 8 else 0.0          # dawn phenomenon
        activity = 0.0015 * steps_per_slot[i]          # walking lowers glucose
        rise = carb_input[i] * 2.0 * ir                # mg/dL per gram absorbed
        cur = cur + rise + dawn - activity - clr * (cur - basal)
        if noise:
            cur += rng.normal(0, 1.2)
        cur = float(np.clip(cur, 55, 400))
        g[i] = cur
    return g
