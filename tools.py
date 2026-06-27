"""
tools.py
The callable tools the investigation agent can invoke.

Each tool is a plain Python function returning a JSON-serializable dict. They
operate over the synthetic claims dataset plus small mock reference tables that
stand in for real payment-policy / coding references. In a production system
these would be database lookups and real policy services; for a POC they are
intentionally tiny and local.
"""

import pandas as pd

# --- mock reference data (consistent with generate_data.py) -------------------

PROCEDURES = {
    "99213": {"desc": "Office visit, established patient, low complexity",
              "price": 110, "max_units": 1, "compatible_dx": {"E11", "I10", "J06", "M54"}},
    "99214": {"desc": "Office visit, established patient, moderate complexity",
              "price": 175, "max_units": 1, "compatible_dx": {"E11", "I10", "J44", "M54"}},
    "93000": {"desc": "Electrocardiogram (ECG), routine",
              "price": 60,  "max_units": 1, "compatible_dx": {"I10", "I48", "R00"}},
    "80053": {"desc": "Comprehensive metabolic panel",
              "price": 45,  "max_units": 1, "compatible_dx": {"E11", "E78", "N18"}},
    "71046": {"desc": "Chest X-ray, 2 views",
              "price": 90,  "max_units": 1, "compatible_dx": {"J18", "J44", "J06"}},
    "20610": {"desc": "Arthrocentesis / major joint injection",
              "price": 230, "max_units": 2, "compatible_dx": {"M17", "M25", "M54"}},
    "97110": {"desc": "Therapeutic exercise, per 15 min",
              "price": 55,  "max_units": 4, "compatible_dx": {"M54", "M25", "S83"}},
}

POLICY_KB = {
    "units": ("CMS Medically Unlikely Edits (MUE) cap the units of service per "
              "line per day. Office/E&M visits and single-session diagnostics are "
              "generally limited to 1 unit/day; claims exceeding the cap are denied "
              "or reviewed."),
    "code_compatibility": ("A procedure must be supported by a medically appropriate "
                           "diagnosis. Procedure/diagnosis pairs outside accepted "
                           "clinical guidelines are flagged for clinical validation."),
    "duplicate": ("Claims with identical provider, member, procedure code, and date "
                  "of service are treated as duplicates and are subject to denial of "
                  "the second occurrence."),
    "overbilling": ("Billed amounts materially above the contracted or typical allowed "
                    "amount for a code are reviewed for upcoding or unbundling."),
}

CLAIMS: pd.DataFrame | None = None


def load_claims(path: str = "claims.csv") -> pd.DataFrame:
    global CLAIMS
    CLAIMS = pd.read_csv(path, dtype={"procedure_code": str, "diagnosis_code": str,
                                      "place_of_service": str})
    return CLAIMS


def _df() -> pd.DataFrame:
    if CLAIMS is None:
        load_claims()
    return CLAIMS


# --- tools -------------------------------------------------------------------

def get_claim_details(claim_id: int) -> dict:
    row = _df()[_df()["claim_id"] == int(claim_id)]
    if row.empty:
        return {"error": f"claim {claim_id} not found"}
    r = row.iloc[0]
    return {
        "claim_id": int(r["claim_id"]), "provider_id": r["provider_id"],
        "member_id": r["member_id"], "procedure_code": r["procedure_code"],
        "diagnosis_code": r["diagnosis_code"], "units": int(r["units"]),
        "place_of_service": str(r["place_of_service"]),
        "service_date": r["service_date"], "billed_amount": float(r["billed_amount"]),
    }


def check_code_compatibility(procedure_code: str, diagnosis_code: str) -> dict:
    ref = PROCEDURES.get(str(procedure_code))
    if not ref:
        return {"known_procedure": False}
    compatible = diagnosis_code[:3] in ref["compatible_dx"]
    return {
        "known_procedure": True, "procedure_desc": ref["desc"],
        "diagnosis_code": diagnosis_code, "compatible": compatible,
        "note": ("diagnosis supports this procedure" if compatible
                 else "diagnosis does NOT clinically support this procedure"),
    }


def check_units_policy(procedure_code: str, units: int) -> dict:
    ref = PROCEDURES.get(str(procedure_code))
    if not ref:
        return {"known_procedure": False}
    within = int(units) <= ref["max_units"]
    return {
        "known_procedure": True, "billed_units": int(units),
        "policy_max_units": ref["max_units"], "within_policy": within,
        "note": ("units within MUE policy" if within
                 else f"units exceed MUE cap of {ref['max_units']}/day"),
    }


def check_duplicates(claim_id: int) -> dict:
    df = _df()
    row = df[df["claim_id"] == int(claim_id)]
    if row.empty:
        return {"error": f"claim {claim_id} not found"}
    r = row.iloc[0]
    matches = df[(df["provider_id"] == r["provider_id"]) &
                 (df["member_id"] == r["member_id"]) &
                 (df["procedure_code"] == r["procedure_code"]) &
                 (df["service_date"] == r["service_date"])]
    other = [int(c) for c in matches["claim_id"] if int(c) != int(claim_id)]
    return {"duplicate_found": len(other) > 0, "matching_claim_ids": other,
            "note": ("possible duplicate billing" if other else "no duplicates found")}


def get_provider_history(provider_id: str) -> dict:
    df = _df()
    pc = df[df["provider_id"] == provider_id]
    if pc.empty:
        return {"error": f"provider {provider_id} not found"}
    peer_avg = df["billed_amount"].mean()
    prov_avg = pc["billed_amount"].mean()
    return {
        "provider_id": provider_id, "total_claims": int(len(pc)),
        "avg_billed_amount": round(float(prov_avg), 2),
        "peer_avg_billed_amount": round(float(peer_avg), 2),
        "ratio_to_peers": round(float(prov_avg / peer_avg), 2),
        "top_codes": pc["procedure_code"].value_counts().head(3).to_dict(),
        "note": ("billing well above peer average" if prov_avg > 1.5 * peer_avg
                 else "billing in normal range vs peers"),
    }


def lookup_payment_policy(topic: str) -> dict:
    topic = str(topic).lower()
    for key, text in POLICY_KB.items():
        if key in topic or topic in key:
            return {"topic": key, "policy": text}
    return {"topic": topic, "policy": "No specific policy found; apply clinical judgment."}


# registry used by the agent
TOOL_FUNCS = {
    "get_claim_details": get_claim_details,
    "check_code_compatibility": check_code_compatibility,
    "check_units_policy": check_units_policy,
    "check_duplicates": check_duplicates,
    "get_provider_history": get_provider_history,
    "lookup_payment_policy": lookup_payment_policy,
}
