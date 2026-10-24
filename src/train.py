"""
Train the GlucoTwin spike predictor.

- Patient-wise split (test patients are never seen in training)
- Model: gradient-boosted trees (scikit-learn HistGradientBoostingClassifier)
- Metrics: AUROC, AUPRC, precision/recall, and warning lead time
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)

from features import CURRENT_MAX, FEATURES, LABELS, SPIKE_MGDL, build_all
from physiology import STEP_MIN

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
THRESHOLD = 0.5


def lead_times(df, proba, threshold):
    """For each spike in test patients: how early did the first alert fire?"""
    out = []
    df = df.assign(p=proba)
    for _, d in df.groupby("patient_id"):
        g = d["glucose"].values
        p = d["p"].values
        cross = np.where((g[1:] > SPIKE_MGDL) & (g[:-1] <= SPIKE_MGDL))[0] + 1
        # Count a spike only if glucose was below the line for the previous 30 min
        cross = [c for c in cross if c >= 6 and (g[c - 6:c] <= SPIKE_MGDL).all()]
        for c in cross:
            window = slice(max(c - 24, 0), c)
            alerts = np.where(p[window] >= threshold)[0]
            if len(alerts):
                out.append((c - (window.start + alerts[0])) * STEP_MIN)
            else:
                out.append(0)
    return np.array(out)


def main():
    os.makedirs(MODELS, exist_ok=True)
    ehr = pd.read_csv(os.path.join(DATA, "ehr.csv"))
    wear = pd.read_csv(os.path.join(DATA, "wearables.csv"))
    meals = pd.read_csv(os.path.join(DATA, "meals.csv"))

    feats = build_all(wear, ehr, meals).dropna(subset=FEATURES + ["spike_2h"])

    pids = ehr["patient_id"].values
    rng = np.random.default_rng(7)
    test_ids = set(rng.choice(pids, size=15, replace=False))
    is_test = feats["patient_id"].isin(test_ids)

    eligible = feats["glucose"] < CURRENT_MAX
    train = feats[~is_test & eligible]
    test_all = feats[is_test]
    test = test_all[test_all["glucose"] < CURRENT_MAX]

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=31, random_state=0)
    model.fit(train[FEATURES], train["spike_2h"].astype(int))

    p = model.predict_proba(test[FEATURES])[:, 1]
    y = test["spike_2h"].astype(int)
    pred = (p >= THRESHOLD).astype(int)

    # Lead time is measured on the full test timeline
    p_all = model.predict_proba(test_all[FEATURES])[:, 1]
    lt = lead_times(test_all, p_all, THRESHOLD)

    metrics = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_patients": int(len(pids) - len(test_ids)), "test_patients": len(test_ids),
        "spike_rate_test": round(float(y.mean()), 3),
        "auroc": round(float(roc_auc_score(y, p)), 3),
        "auprc": round(float(average_precision_score(y, p)), 3),
        "precision": round(float(precision_score(y, pred)), 3),
        "recall": round(float(recall_score(y, pred)), 3),
        "spikes_in_test": int(len(lt)),
        "spikes_warned_30min_ahead": round(float((lt >= 30).mean()), 3),
        "median_lead_time_min": float(np.median(lt)) if len(lt) else 0.0,
    }

    sample = test.sample(min(5000, len(test)), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample["spike_2h"].astype(int),
                                 scoring="roc_auc", n_repeats=3, random_state=0)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .assign(label=lambda d: d.feature.map(LABELS))
                  .sort_values("importance", ascending=False))

    joblib.dump({"model": model, "features": FEATURES, "threshold": THRESHOLD,
                 "test_patients": sorted(test_ids),
                 "medians": train[FEATURES].median().to_dict()},
                os.path.join(MODELS, "glucotwin.joblib"))
    importance.to_csv(os.path.join(MODELS, "feature_importance.csv"), index=False)
    with open(os.path.join(MODELS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    print("\nTop drivers:\n", importance.head(8)[["label", "importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
