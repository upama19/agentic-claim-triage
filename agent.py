"""
agent.py
Stage 2 of the pipeline: agentic investigation.

For a single flagged claim, the agent autonomously decides which tools to call,
reasons over the results, and returns a structured triage decision:

    {triage, confidence, rationale, trace}

triage is one of:
    AUTO_APPROVE                - looks legitimate, no action needed
    ROUTE_TO_REVIEW             - suspicious, send to a human analyst
    HIGH_PRIORITY_INVESTIGATE   - strong FWA signal, escalate

Two execution modes:
  * LIVE   - uses the Anthropic API (set ANTHROPIC_API_KEY). The model chooses
             tools via native tool-use.
  * OFFLINE- a deterministic investigator that calls the same tools with simple
             heuristics. Lets the demo/video run with no key or network.
"""

import json
import os

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads ANTHROPIC_API_KEY from a local .env file if present
except ImportError:
    pass  # dotenv is optional; the env var still works without it

from tools import TOOL_FUNCS, get_claim_details

# Set this to  model which Anthropic account can access.
MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are a payment-integrity investigation agent for a health plan.
A statistical model has flagged a medical claim as anomalous. Your job is to
investigate it using the available tools, then decide how to triage it.

Investigate methodically: pull the claim details first, then check whatever is
relevant (units policy, code/diagnosis compatibility, duplicates, provider
history, payment policy). Do not guess - base your conclusion on tool results.

When finished, respond with ONLY a JSON object (no prose) of the form:
{"triage": "...", "confidence": 0.0-1.0, "rationale": "one or two sentences"}
where triage is one of AUTO_APPROVE, ROUTE_TO_REVIEW, HIGH_PRIORITY_INVESTIGATE.
Never output a final denial - the most severe action you may recommend is
HIGH_PRIORITY_INVESTIGATE, leaving the final payment decision to a human."""

# Anthropic tool schemas
TOOL_SCHEMAS = [
    {
        "name": "get_claim_details",
        "description": "Get all fields for a claim by id.",
        "input_schema": {
            "type": "object",
            "properties": {"claim_id": {"type": "integer"}},
            "required": ["claim_id"],
        },
    },
    {
        "name": "check_code_compatibility",
        "description": "Check whether a diagnosis supports a procedure code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "procedure_code": {"type": "string"},
                "diagnosis_code": {"type": "string"},
            },
            "required": ["procedure_code", "diagnosis_code"],
        },
    },
    {
        "name": "check_units_policy",
        "description": "Check billed units against the MUE per-day cap for a procedure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "procedure_code": {"type": "string"},
                "units": {"type": "integer"},
            },
            "required": ["procedure_code", "units"],
        },
    },
    {
        "name": "check_duplicates",
        "description": "Check whether a claim duplicates another (same provider/member/code/date).",
        "input_schema": {
            "type": "object",
            "properties": {"claim_id": {"type": "integer"}},
            "required": ["claim_id"],
        },
    },
    {
        "name": "get_provider_history",
        "description": "Get a provider's billing stats vs peers.",
        "input_schema": {
            "type": "object",
            "properties": {"provider_id": {"type": "string"}},
            "required": ["provider_id"],
        },
    },
    {
        "name": "lookup_payment_policy",
        "description": "Look up payment policy text for a topic (units, code_compatibility, duplicate, overbilling).",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]


def _run_tool(name, args, trace):
    result = TOOL_FUNCS[name](**args)
    trace.append({"tool": name, "input": args, "result": result})
    return result


# --- LIVE mode ---------------------------------------------------------------


def investigate_live(claim_id: int, max_steps: int = 8) -> dict:
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    trace = []
    messages = [
        {
            "role": "user",
            "content": f"Investigate flagged claim {claim_id} and triage it.",
        }
    ]
    for _ in range(max_steps):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = _run_tool(block.name, block.input, trace)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(out),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
            decision = _parse_decision(text)
            decision["trace"] = trace
            return decision
    return {
        "triage": "ROUTE_TO_REVIEW",
        "confidence": 0.5,
        "rationale": "Investigation did not converge; routing to a human.",
        "trace": trace,
    }


def _parse_decision(text: str) -> dict:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        d = json.loads(text[start:end])
        d.setdefault("triage", "ROUTE_TO_REVIEW")
        d.setdefault("confidence", 0.5)
        d.setdefault("rationale", "")
        return d
    except Exception:
        return {
            "triage": "ROUTE_TO_REVIEW",
            "confidence": 0.5,
            "rationale": text[:200] or "Unparseable model output.",
        }


# --- OFFLINE mode ------------------------------------------------------------


def investigate_offline(claim_id: int) -> dict:
    """Deterministic investigator using the same tools and simple heuristics."""
    trace = []
    c = _run_tool("get_claim_details", {"claim_id": int(claim_id)}, trace)
    if "error" in c:
        return {
            "triage": "ROUTE_TO_REVIEW",
            "confidence": 0.5,
            "rationale": "Claim not found.",
            "trace": trace,
        }

    units = _run_tool(
        "check_units_policy",
        {"procedure_code": c["procedure_code"], "units": c["units"]},
        trace,
    )
    compat = _run_tool(
        "check_code_compatibility",
        {"procedure_code": c["procedure_code"], "diagnosis_code": c["diagnosis_code"]},
        trace,
    )
    dupe = _run_tool("check_duplicates", {"claim_id": int(claim_id)}, trace)
    prov = _run_tool("get_provider_history", {"provider_id": c["provider_id"]}, trace)

    reasons, severity = [], 0
    if units.get("within_policy") is False:
        reasons.append(units["note"])
        severity += 2
    if compat.get("compatible") is False:
        reasons.append(compat["note"])
        severity += 2
    if dupe.get("duplicate_found"):
        reasons.append(dupe["note"])
        severity += 1
    if prov.get("ratio_to_peers", 1) > 1.5:
        reasons.append(prov["note"])
        severity += 1

    if severity >= 2:
        triage, conf = "HIGH_PRIORITY_INVESTIGATE", min(0.95, 0.6 + 0.12 * severity)
    elif severity == 1:
        triage, conf = "ROUTE_TO_REVIEW", 0.7
    else:
        triage, conf = "AUTO_APPROVE", 0.8
    rationale = (
        "; ".join(reasons)
        if reasons
        else "No policy violations found on investigation; appears legitimate."
    )
    return {
        "triage": triage,
        "confidence": round(conf, 2),
        "rationale": rationale.capitalize(),
        "trace": trace,
    }


def investigate(claim_id: int, live: bool | None = None) -> dict:
    """Auto-selects live mode if a key is present, unless overridden."""
    if live is None:
        live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if live:
        try:
            return investigate_live(claim_id)
        except Exception as e:
            out = investigate_offline(claim_id)
            out["rationale"] += f"  [fell back to offline mode: {e}]"
            return out
    return investigate_offline(claim_id)


if __name__ == "__main__":
    import sys
    from tools import load_claims

    load_claims()
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 100320
    result = investigate(cid, live=False)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
    print(f"\nTool calls made ({len(result['trace'])}):")
    for step in result["trace"]:
        print(f"  - {step['tool']}({step['input']})")
