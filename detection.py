"""
detection.py
Stage 1 of the pipeline: unsupervised anomaly detection.

We engineer a few numeric features from each claim and fit an IsolationForest.
The model assigns every claim an anomaly score; the most anomalous claims are
"flagged" and passed to the agentic investigation layer (Stage 2).

Design note: the ML model deliberately does NOT try to explain *why* a claim is
suspicious - it only surfaces statistical outliers. Explaining and triaging the
"why" is the job of the agent. That separation is the core thesis of the project:
ML for recall, agent for reasoning + precision.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# typical price per code, mirrored from generate_data.py (kept small + local for the POC)
TYPICAL_PRICE = {
    "99213": 110, "99214": 175, "93000": 60, "80053": 45,
    "71046": 90, "20610": 230, "97110": 55,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["typical_price"] = out["procedure_code"].map(TYPICAL_PRICE).fillna(out["billed_amount"])
    out["price_ratio"] = out["billed_amount"] / out["typical_price"]
    # provider-level signal: how much this provider's billing deviates from peers
    prov_avg = out.groupby("provider_id")["price_ratio"].transform("mean")
    out["provider_price_ratio"] = prov_avg
    # duplicate signal: identical provider/member/code/date seen more than once
    dup_key = ["provider_id", "member_id", "procedure_code", "service_date"]
    out["dup_count"] = out.groupby(dup_key)["claim_id"].transform("count")
    return out


FEATURES = ["billed_amount", "units", "price_ratio", "provider_price_ratio", "dup_count"]


def detect(df: pd.DataFrame, contamination: float = 0.15) -> pd.DataFrame:
    feats = engineer_features(df)
    X = feats[FEATURES].to_numpy()
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X)
    # score_samples: higher = more normal. Invert so higher = more anomalous.
    raw = model.score_samples(X)
    feats["anomaly_score"] = (-raw - (-raw).min()) / ((-raw).max() - (-raw).min())
    feats["flagged"] = model.predict(X) == -1
    return feats.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("claims.csv")
    scored = detect(df)
    flagged = scored[scored["flagged"]]
    print(f"Flagged {len(flagged)} of {len(scored)} claims for investigation\n")

    # quick self-evaluation against the hidden ground-truth labels
    truth_anomalous = scored["_label"] != "normal"
    caught = (scored["flagged"] & truth_anomalous).sum()
    total_anom = truth_anomalous.sum()
    false_alarms = (scored["flagged"] & ~truth_anomalous).sum()
    print(f"Recall on injected anomalies: {caught}/{total_anom} = {caught/total_anom:.0%}")
    print(f"False positives among flags:  {false_alarms}/{len(flagged)} "
          f"= {false_alarms/len(flagged):.0%}  <- this is exactly what the agent triages down\n")

    print("Top 8 most anomalous claims:")
    cols = ["claim_id", "provider_id", "procedure_code", "diagnosis_code",
            "units", "billed_amount", "anomaly_score", "_label"]
    print(scored[cols].head(8).to_string(index=False))
