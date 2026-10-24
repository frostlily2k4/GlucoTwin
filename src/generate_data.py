"""
Generate synthetic data for the GlucoTwin proof-of-concept.

Outputs (in data/):
  ehr.csv        - one row per patient (static / historical data)
  wearables.csv  - 5-minute time series per patient (dynamic / real-time data)
  meals.csv      - meal log per patient

All data is synthetic. No real patient data is used (DPDP Act / HIPAA safe).
The EHR fields mirror what a Synthea export would give you, so you can swap
in Synthea or MIMIC-IV later without changing the rest of the pipeline.
"""
import os
import numpy as np
import pandas as pd

from physiology import (MEALS, STEP_MIN, STEPS_PER_DAY, patient_params,
                        simulate)

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
N_PATIENTS = 60
N_DAYS = 14
SEED = 42

FIRST_NAMES = ["Arun", "Priya", "Karthik", "Lakshmi", "Suresh", "Meena",
               "Ravi", "Divya", "Senthil", "Anitha", "Vijay", "Kavitha",
               "Murali", "Revathi", "Ganesh", "Deepa"]


def make_ehr(rng):
    rows = []
    for pid in range(1, N_PATIENTS + 1):
        age = int(rng.integers(32, 72))
        sex = rng.choice(["M", "F"])
        bmi = round(float(rng.normal(25.5, 3.2)), 1)
        hba1c = round(float(np.clip(rng.normal(7.4, 1.0), 5.8, 10.5)), 1)
        rows.append({
            "patient_id": f"P{pid:03d}",
            "name": f"{rng.choice(FIRST_NAMES)} (synthetic)",
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "hba1c": hba1c,
            "fasting_glucose": round(28.7 * hba1c - 46.7 + rng.normal(-15, 8)),
            "ldl": int(rng.normal(120, 25)),
            "systolic_bp": int(rng.normal(132, 12)),
            "years_since_diagnosis": int(rng.integers(0, 15)),
            "family_history_t2d": int(rng.random() < 0.6),
            "hypertension_dx": int(rng.random() < 0.4),
            "on_metformin": int(rng.random() < 0.7),
            "tcf7l2_risk_allele": int(rng.random() < 0.3),  # genetic marker
            "diet_pattern": rng.choice(["rice_heavy", "mixed", "wheat_heavy"]),
        })
    return pd.DataFrame(rows)


def pick_meal(slot, diet, rng):
    if slot == "breakfast":
        opts = ["idli_sambar", "dosa_chutney", "pongal"]
    elif slot == "tea":
        opts = ["tea_biscuits"]
    else:
        if diet == "rice_heavy":
            opts = ["rice_sambar_rasam", "biryani", "curd_rice", "rice_sambar_rasam"]
        elif diet == "wheat_heavy":
            opts = ["chapati_dal", "chapati_dal", "millet_meal", "rice_sambar_rasam"]
        else:
            opts = ["rice_sambar_rasam", "chapati_dal", "millet_meal", "biryani"]
    return rng.choice(opts)


def simulate_patient(ehr_row, rng):
    params = patient_params(ehr_row)
    n = N_DAYS * STEPS_PER_DAY
    steps = np.zeros(n)
    asleep = np.zeros(n, dtype=int)
    sleep_hours_by_day = []
    meals, meal_log = {}, []

    for d in range(N_DAYS):
        base = d * STEPS_PER_DAY
        # Sleep from previous night: wake around 6:30
        sleep_h = float(np.clip(rng.normal(6.8, 1.0), 4.0, 9.0))
        sleep_hours_by_day.append(sleep_h)
        wake = base + int((6.5 + rng.normal(0, 0.4)) * 60 / STEP_MIN)
        sleep_start = wake - int(sleep_h * 60 / STEP_MIN)
        asleep[max(sleep_start, 0):wake] = 1

        # Daytime activity
        for i in range(wake, base + STEPS_PER_DAY - 12):
            if i < n:
                steps[i] = max(0, rng.normal(25, 25)) if rng.random() < 0.5 else 0

        # Meals: South Indian timing, late dinners common
        times = {
            "breakfast": 8.0 + rng.normal(0, 0.5),
            "lunch": 13.5 + rng.normal(0, 0.5),
            "dinner": 21.0 + rng.normal(0, 0.75),
        }
        if rng.random() < 0.7:
            times["tea"] = 17.0 + rng.normal(0, 0.4)
        for slot, hr in times.items():
            idx = base + int(hr * 60 / STEP_MIN)
            meal = pick_meal(slot, ehr_row["diet_pattern"], rng)
            carbs, speed = MEALS[meal]
            carbs = carbs * rng.uniform(0.85, 1.2)
            meals[idx] = (carbs, speed)
            meal_log.append({"patient_id": ehr_row["patient_id"], "step": idx,
                             "meal": meal, "carbs_g": round(carbs, 1)})
            # Post-meal walk sometimes
            if slot in ("lunch", "dinner") and rng.random() < 0.3:
                walk = int(rng.integers(3, 6))
                steps[idx + 4: idx + 4 + walk] += 450

    # Simulate glucose day by day so sleep debt affects each day
    glucose = np.zeros(n)
    cur = params["basal"]
    for d in range(N_DAYS):
        s, e = d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY
        day_meals = {k - s: v for k, v in meals.items() if s <= k < e}
        g = simulate(params, STEPS_PER_DAY, day_meals, steps[s:e],
                     sleep_debt_hours=7.5 - sleep_hours_by_day[d],
                     start_glucose=cur, start_step_of_day=0, rng=rng)
        glucose[s:e] = g
        cur = g[-1]

    # Heart rate and HRV
    resting_hr = 68 + (ehr_row["bmi"] - 24) * 0.8 + rng.normal(0, 4)
    hr = resting_hr + steps * 0.05 + rng.normal(0, 2.5, n) - asleep * 8
    day_idx = np.arange(n) // STEPS_PER_DAY
    sleep_debt = np.array([7.5 - sleep_hours_by_day[d] for d in day_idx])
    hr += np.clip(sleep_debt, 0, None) * 1.5
    hrv = 42 - (ehr_row["age"] - 40) * 0.3 - np.clip(sleep_debt, 0, None) * 3 \
        + asleep * 10 + rng.normal(0, 4, n)

    # Sleep stage (only meaningful while asleep)
    stages = np.where(asleep == 1,
                      rng.choice(["light", "deep", "rem"], n, p=[0.55, 0.2, 0.25]),
                      "awake")

    ts = pd.date_range("2026-01-01", periods=n, freq=f"{STEP_MIN}min")
    df = pd.DataFrame({
        "patient_id": ehr_row["patient_id"],
        "timestamp": ts,
        "step": np.arange(n),
        "glucose_mgdl": glucose.round(1),
        "heart_rate": hr.round(1),
        "hrv_ms": np.clip(hrv, 8, None).round(1),
        "steps": steps.round(0).astype(int),
        "sleep_stage": stages,
        "last_night_sleep_h": np.array([sleep_hours_by_day[d] for d in day_idx]).round(2),
    })
    return df, meal_log


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ehr = make_ehr(rng)
    frames, all_meals = [], []
    for _, row in ehr.iterrows():
        df, ml = simulate_patient(row.to_dict(), rng)
        frames.append(df)
        all_meals.extend(ml)
    wear = pd.concat(frames, ignore_index=True)
    ehr.to_csv(os.path.join(OUT, "ehr.csv"), index=False)
    wear.to_csv(os.path.join(OUT, "wearables.csv"), index=False)
    pd.DataFrame(all_meals).to_csv(os.path.join(OUT, "meals.csv"), index=False)
    print(f"EHR: {len(ehr)} patients | wearables: {len(wear):,} rows | "
          f"meals: {len(all_meals):,}")
    print(f"Glucose range: {wear.glucose_mgdl.min():.0f}-{wear.glucose_mgdl.max():.0f} mg/dL, "
          f"% time >180: {(wear.glucose_mgdl > 180).mean():.1%}")


if __name__ == "__main__":
    main()
