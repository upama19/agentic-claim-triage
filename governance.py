"""
governance.py
Stage 3: governance, safety, and auditability.

This layer sits between the agent's recommendation and any action. It encodes
three safety rules that reflect responsible-AI practice for agentic systems in
a regulated (healthcare payment) setting:

  1. Human-in-the-loop on consequential actions: the agent can never auto-deny a
     claim. Any high-severity recommendation is routed to a human analyst.
  2. Confidence gating: low-confidence decisions are escalated to a human rather
     than acted on automatically.
  3. Auditability: every tool call and decision is recorded so the reasoning
     behind each triage is fully traceable after the fact.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

CONFIDENCE_FLOOR = 0.60  # below this, always send to a human


@dataclass
class AuditLog:
    entries: list = field(default_factory=list)

    def record(self, claim_id, decision, routing):
        self.entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "claim_id": claim_id,
            "agent_triage": decision.get("triage"),
            "confidence": decision.get("confidence"),
            "final_routing": routing["final_routing"],
            "human_required": routing["human_required"],
            "governance_notes": routing["notes"],
            "tool_calls": [s["tool"] for s in decision.get("trace", [])],
        })

    def to_list(self):
        return self.entries


def apply_governance(decision: dict) -> dict:
    """Map an agent decision to a final, safety-checked routing outcome."""
    triage = decision.get("triage", "ROUTE_TO_REVIEW")
    confidence = float(decision.get("confidence", 0.5))
    notes = []

    human_required = False
    if triage == "HIGH_PRIORITY_INVESTIGATE":
        human_required = True
        notes.append("High-severity recommendation: human analyst required (no auto-deny).")
    if confidence < CONFIDENCE_FLOOR:
        human_required = True
        notes.append(f"Confidence {confidence:.2f} below floor {CONFIDENCE_FLOOR}: escalated to human.")
    if triage == "ROUTE_TO_REVIEW":
        human_required = True
        notes.append("Agent routed to review: queued for human analyst.")

    if triage == "AUTO_APPROVE" and not human_required:
        final = "AUTO_APPROVED (no payment-integrity action)"
    elif triage == "HIGH_PRIORITY_INVESTIGATE":
        final = "ESCALATED to senior analyst queue"
    else:
        final = "QUEUED for human review"

    if not notes:
        notes.append("Auto-approved within confidence and severity thresholds.")

    return {"final_routing": final, "human_required": human_required, "notes": notes}


if __name__ == "__main__":
    from tools import load_claims
    from agent import investigate
    load_claims()
    log = AuditLog()
    for cid in [100320, 100228]:
        d = investigate(cid, live=False)
        r = apply_governance(d)
        log.record(cid, d, r)
        print(f"claim {cid}: {d['triage']} (conf {d['confidence']}) "
              f"-> {r['final_routing']} | human_required={r['human_required']}")
    print("\nAudit log:")
    import json
    print(json.dumps(log.to_list(), indent=2))
