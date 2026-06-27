# Agentic FWA Triage

A hybrid **machine-learning + agentic-AI** prototype for healthcare **payment integrity**.

A statistical model flags anomalous medical claims; an autonomous LLM agent then
**investigates** each flagged claim using callable tools, reasons over the evidence,
and **triages** it — all behind a human-in-the-loop **governance** layer that keeps a
person in control of consequential decisions and logs every step for auditability.

> Built as a proof-of-concept. All claims data is **synthetic** — no real PHI.

## Why this design

Healthcare payers lose large sums to fraud, waste, and abuse (FWA). Pure
anomaly-detection models surface a lot of suspicious claims but generate **many
false positives**, overwhelming human analysts ("alert fatigue"). Pure LLM
approaches reason fluently but hallucinate and don't scale over structured claims
data.

This project's thesis: combine them. Use ML for **recall** (catch the outliers)
and an agent for **reasoning and precision** (investigate the *why*, triage the
noise, and escalate only what matters — with a human always in the loop).

On the synthetic dataset, Stage 1 catches ~62% of injected anomalies but ~37% of
its flags are false positives. Stage 2 exists to triage that 37% down by
investigating each flag rather than dumping all of them on an analyst.

## Architecture

```
                 ┌─────────────────┐
 claims.csv ───▶ │  Stage 1:       │  IsolationForest over engineered
                 │  Detection      │  features → anomaly score + flag
                 └────────┬────────┘
                          │  flagged claims
                          ▼
                 ┌─────────────────┐   tools:
                 │  Stage 2:       │   • get_claim_details
                 │  Agent          │   • check_code_compatibility
                 │  (tool-calling) │   • check_units_policy
                 │                 │   • check_duplicates
                 │                 │   • get_provider_history
                 └────────┬────────┘   • lookup_payment_policy
                          │  triage + confidence + rationale + trace
                          ▼
                 ┌─────────────────┐  • never auto-deny (human-in-the-loop)
                 │  Stage 3:       │  • confidence gating
                 │  Governance     │  • full audit log
                 └─────────────────┘
```

## Files

| File | Role |
|------|------|
| `generate_data.py` | Builds the synthetic claims dataset with injected FWA patterns |
| `detection.py`     | Stage 1 — IsolationForest anomaly detection |
| `tools.py`         | The agent's callable tools + mock policy references |
| `agent.py`         | Stage 2 — Claude tool-calling loop (+ offline fallback) |
| `governance.py`    | Stage 3 — human-in-the-loop routing + audit log |
| `app.py`           | Streamlit demo UI |

## Quickstart

```bash
pip install -r requirements.txt
python generate_data.py          # writes claims.csv
streamlit run app.py             # launches the demo
```

The app runs in **offline mode** with no API key (deterministic investigator, so
the demo always works). For the **live agent**, set a key and pick "Live" in the
sidebar:

```bash
export ANTHROPIC_API_KEY=sk-...   # Windows: set ANTHROPIC_API_KEY=sk-...
```

Set the `MODEL` constant in `agent.py` to whatever model your account can access.

Run individual stages from the CLI to see them in isolation:

```bash
python detection.py     # detection + recall/false-positive breakdown
python agent.py 100320  # investigate one claim
python governance.py    # full pipeline + audit log
```

## Governance & safety

Because this is an agent making decisions in a regulated domain, three safeguards
are built in: the agent can never auto-deny (its most severe action is to escalate
to a human), low-confidence decisions are gated to human review, and every tool
call and decision is written to an auditable log.

## Limitations & future work

Synthetic data and tiny mock policy tables stand in for real claims and coding
references; the detector is a simple unsupervised baseline. Natural extensions:
real CPT/ICD reference services, a richer policy retrieval layer, supervised
detection with feedback from analyst dispositions, and formal agent evaluation
(decision accuracy, escalation precision) against labeled outcomes.
