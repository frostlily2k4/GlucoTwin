"""
Fuse the two data streams into model features.

Static stream  (EHR)       -> patient-level features, repeated on every row
Dynamic stream (wearables) -> rolling/lag features computed ONLY from the past

Target: will glucose cross 180 mg/dL at any point in the next 2 hours?
We only predict from moments when the patient is currently below 160 mg/dL,
so the model is forecasting a new spike, not describing one already happening.
"""
import numpy as np
import pandas as pd

from physiology import STEP_MIN, STEPS_PER_DAY

HORIZON_STEPS = 120 // STEP_MIN      # 2 hours ahead
SPIKE_MGDL = 180
CURRENT_MAX = 160

EHR_FEATURES = ["age", "bmi", "hba1c", "fasting_glucose", "ldl", "systolic_bp",
                "years_since_diagnosis", "family_history_t2d", "hypertension_dx",
                "on_metformin", "tcf7l2_risk_allele", "diet_rice_heavy"]

WEARABLE_FEATURES = [
    "glucose", "glucose_lag_15", "glucose_lag_30", "glucose_lag_60",
    "slope_15", "slope_30", "glucose_mean_3h", "glucose_std_3h",
    "hr_mean_30", "hrv_mean_60", "steps_30", "steps_120",
    "last_night_sleep_h", "hour_sin", "hour_cos",
    "mins_since_meal", "carbs_last_meal", "carbs_last_3h",
    "mins_to_usual_meal",
]

FEATURES = WEARABLE_FEATURES + EHR_FEATURES

# Plain-language names for the doctor dashboard
LABELS = {
    "glucose": "Current glucose", "glucose_lag_15": "Glucose 15 min ago",
    "glucose_lag_30": "Glucose 30 min ago", "glucose_lag_60": "Glucose 1 h ago",
    "slope_15": "Glucose trend (15 min)", "slope_30": "Glucose trend (30 min)",
    "glucose_mean_3h": "Average glucose (3 h)", "glucose_std_3h": "Glucose variability (3 h)",
    "hr_mean_30": "Heart rate (30 min)", "hrv_mean_60": "Heart rate variability",
    "steps_30": "Steps (30 min)", "steps_120": "Steps (2 h)",
    "last_night_sleep_h": "Last night's sleep", "hour_sin": "Time of day",
    "hour_cos": "Time of day", "mins_since_meal": "Time since last meal",
    "carbs_last_meal": "Carbs in last meal", "carbs_last_3h": "Carbs in last 3 h",
    "mins_to_usual_meal": "Time to usual meal", "age": "Age", "bmi": "BMI",
    "hba1c": "HbA1c", "fasting_glucose": "Fasting glucose", "ldl": "LDL",
    "systolic_bp": "Systolic BP", "years_since_diagnosis": "Years since diagnosis",
    "family_history_t2d": "Family history of T2D", "hypertension_dx": "Hypertension",
    "on_metformin": "On metformin", "tcf7l2_risk_allele": "TCF7L2 risk allele",
    "diet_rice_heavy": "Rice-heavy diet",
}


def usual_meal_steps(meals_p: pd.DataFrame) -> np.ndarray:
    """A patient's habitual meal times (step-of-day), learnt from their history."""
    sod = meals_p["step"].values % STEPS_PER_DAY
    centres = []
    for lo, hi in [(60, 132), (132, 192), (192, 234), (234, 288)]:  # 5-11, 11-16, 16-19.5, 19.5-24h
        sel = sod[(sod >= lo) & (sod < hi)]
        if len(sel):
            centres.append(int(np.median(sel)))
    return np.array(centres)


def build_features(wear_p: pd.DataFrame, ehr_row: dict, meals_p: pd.DataFrame,
                   with_label: bool = True) -> pd.DataFrame:
    w = wear_p.sort_values("step").reset_index(drop=True)
    g = w["glucose_mgdl"]
    f = pd.DataFrame({"patient_id": w["patient_id"], "step": w["step"],
                      "timestamp": w["timestamp"]})

    f["glucose"] = g
    f["glucose_lag_15"] = g.shift(3)
    f["glucose_lag_30"] = g.shift(6)
    f["glucose_lag_60"] = g.shift(12)
    f["slope_15"] = (g - g.shift(3)) / 15
    f["slope_30"] = (g - g.shift(6)) / 30
    f["glucose_mean_3h"] = g.rolling(36).mean()
    f["glucose_std_3h"] = g.rolling(36).std()
    f["hr_mean_30"] = w["heart_rate"].rolling(6).mean()
    f["hrv_mean_60"] = w["hrv_ms"].rolling(12).mean()
    f["steps_30"] = w["steps"].rolling(6).sum()
    f["steps_120"] = w["steps"].rolling(24).sum()
    f["last_night_sleep_h"] = w["last_night_sleep_h"]

    sod = w["step"] % STEPS_PER_DAY
    hour = sod * STEP_MIN / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Meal features: only meals logged at or before time t
    carbs = np.zeros(len(w))
    meal_idx = meals_p["step"].values
    for s, c in zip(meal_idx, meals_p["carbs_g"].values):
        pos = s - w["step"].iloc[0]
        if 0 <= pos < len(w):
            carbs[pos] += c
    carbs_s = pd.Series(carbs)
    f["carbs_last_3h"] = carbs_s.rolling(36, min_periods=1).sum()
    last_meal_pos = pd.Series(np.where(carbs > 0, np.arange(len(w)), np.nan)).ffill()
    f["mins_since_meal"] = ((np.arange(len(w)) - last_meal_pos) * STEP_MIN).fillna(600).clip(upper=600)
    f["carbs_last_meal"] = pd.Series(np.where(carbs > 0, carbs, np.nan)).ffill().fillna(0)

    usual = usual_meal_steps(meals_p)
    diff = (usual[None, :] - sod.values[:, None]) % STEPS_PER_DAY
    f["mins_to_usual_meal"] = diff.min(axis=1) * STEP_MIN

    for col in EHR_FEATURES:
        if col == "diet_rice_heavy":
            f[col] = int(ehr_row["diet_pattern"] == "rice_heavy")
        else:
            f[col] = ehr_row[col]

    if with_label:
        future_max = g[::-1].rolling(HORIZON_STEPS, min_periods=1).max()[::-1].shift(-1)
        f["future_max"] = future_max
        f["spike_2h"] = (future_max > SPIKE_MGDL).astype(int)
        f.loc[future_max.isna(), "spike_2h"] = np.nan
    return f


def build_all(wear, ehr, meals, with_label=True):
    out = []
    for _, row in ehr.iterrows():
        pid = row["patient_id"]
        out.append(build_features(wear[wear.patient_id == pid], row.to_dict(),
                                  meals[meals.patient_id == pid], with_label))
    return pd.concat(out, ignore_index=True)
