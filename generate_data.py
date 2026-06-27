"""
generate_data.py
Creates a synthetic medical-claims dataset for the Agentic FWA Triage POC.

The data is intentionally NOT real PHI. It mimics the *shape* of professional
(CMS-1500-style) claims and injects several well-known fraud/waste/abuse (FWA)
patterns so the detection + agent layers have something meaningful to find.

Injected patterns:
  1. Overbilling      - billed_amount far above the normal range for the code
  2. Impossible units - e.g. 40 units of a once-per-day procedure
  3. Code mismatch    - procedure incompatible with the diagnosis
  4. Duplicate claim  - same provider/member/code/date billed twice
  5. Upcoding provider- a provider whose claims skew to high-reimbursement codes
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# A tiny mock reference table: CPT-like code -> (typical price, max sane units, compatible ICD prefixes)
PROCEDURES = {
    "99213": {"price": 110, "max_units": 1, "compatible_dx": {"E11", "I10", "J06", "M54"}},  # office visit
    "99214": {"price": 175, "max_units": 1, "compatible_dx": {"E11", "I10", "J44", "M54"}},  # office visit, complex
    "93000": {"price": 60,  "max_units": 1, "compatible_dx": {"I10", "I48", "R00"}},          # ECG
    "80053": {"price": 45,  "max_units": 1, "compatible_dx": {"E11", "E78", "N18"}},          # metabolic panel
    "71046": {"price": 90,  "max_units": 1, "compatible_dx": {"J18", "J44", "J06"}},          # chest x-ray
    "20610": {"price": 230, "max_units": 2, "compatible_dx": {"M17", "M25", "M54"}},          # joint injection
    "97110": {"price": 55,  "max_units": 4, "compatible_dx": {"M54", "M25", "S83"}},          # therapeutic exercise
}
PROC_CODES = list(PROCEDURES.keys())
ALL_DX = ["E11.9", "I10", "J06.9", "M54.5", "J44.9", "I48.0", "R00.0",
          "E78.5", "N18.3", "J18.9", "M17.0", "M25.5", "S83.5"]
POS = ["11", "22", "23", "19"]  # place of service: office, outpatient, ER, off-campus


def _base_claim(claim_id, provider_id):
    code = RNG.choice(PROC_CODES)
    ref = PROCEDURES[code]
    # pick a compatible diagnosis most of the time
    compatible = [d for d in ALL_DX if d[:3] in ref["compatible_dx"]]
    dx = RNG.choice(compatible) if compatible else RNG.choice(ALL_DX)
    price = ref["price"] * RNG.uniform(0.9, 1.15)  # normal jitter
    return {
        "claim_id": claim_id,
        "provider_id": provider_id,
        "member_id": f"M{RNG.integers(1000, 9999)}",
        "procedure_code": code,
        "diagnosis_code": dx,
        "units": 1 if ref["max_units"] == 1 else int(RNG.integers(1, ref["max_units"] + 1)),
        "place_of_service": RNG.choice(POS),
        "service_date": f"2026-0{RNG.integers(1,7)}-{RNG.integers(10,28)}",
        "billed_amount": round(price, 2),
        "_label": "normal",  # ground-truth tag, kept only for our own evaluation
    }


def generate(n_normal=300, seed=42):
    global RNG
    RNG = np.random.default_rng(seed)
    providers = [f"P{1000+i}" for i in range(12)]
    rows = []
    cid = 100000

    # normal claims
    for _ in range(n_normal):
        cid += 1
        rows.append(_base_claim(cid, RNG.choice(providers)))

    # 1. overbilling
    for _ in range(12):
        cid += 1
        r = _base_claim(cid, RNG.choice(providers))
        r["billed_amount"] = round(r["billed_amount"] * RNG.uniform(4, 9), 2)
        r["_label"] = "overbilling"
        rows.append(r)

    # 2. impossible units
    for _ in range(10):
        cid += 1
        r = _base_claim(cid, RNG.choice(providers))
        r["units"] = int(RNG.integers(15, 50))
        r["billed_amount"] = round(r["billed_amount"] * r["units"] * 0.8, 2)
        r["_label"] = "impossible_units"
        rows.append(r)

    # 3. procedure/diagnosis mismatch
    for _ in range(10):
        cid += 1
        r = _base_claim(cid, RNG.choice(providers))
        ref = PROCEDURES[r["procedure_code"]]
        bad = [d for d in ALL_DX if d[:3] not in ref["compatible_dx"]]
        r["diagnosis_code"] = RNG.choice(bad)
        r["_label"] = "code_mismatch"
        rows.append(r)

    # 4. duplicates (clone an existing normal claim)
    base_for_dupes = [r for r in rows if r["_label"] == "normal"][:8]
    for r0 in base_for_dupes:
        cid += 1
        r = dict(r0)
        r["claim_id"] = cid
        r["_label"] = "duplicate"
        rows.append(r)

    # 5. an upcoding provider: P9999 bills mostly the expensive codes at high $
    for _ in range(15):
        cid += 1
        r = _base_claim(cid, "P9999")
        r["procedure_code"] = RNG.choice(["99214", "20610"])
        ref = PROCEDURES[r["procedure_code"]]
        r["billed_amount"] = round(ref["price"] * RNG.uniform(1.8, 3.0), 2)
        r["_label"] = "upcoding_provider"
        rows.append(r)

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("claims.csv", index=False)
    print(f"Wrote claims.csv with {len(df)} claims")
    print("\nLabel distribution:")
    print(df["_label"].value_counts().to_string())
    print("\nSample rows:")
    print(df.drop(columns=["_label"]).head(6).to_string(index=False))
