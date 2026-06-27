"""
app.py
Streamlit demo for the Agentic FWA Triage POC.

Run:  streamlit run app.py

Pipeline shown in the UI:
  Stage 1  IsolationForest flags anomalous claims
  Stage 2  an autonomous agent investigates each flagged claim via tools
  Stage 3  a governance layer applies safety rules and logs every decision
"""

import os
import pandas as pd
import streamlit as st

from tools import load_claims
from detection import detect
from agent import investigate
from governance import apply_governance, AuditLog

st.set_page_config(page_title="Agentic FWA Triage", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- styling -----
st.markdown(
    """
<style>
:root {
  --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg-soft:#f8fafc;
  --brand:#4338ca; --brand2:#6d28d9;
  --green:#16a34a; --amber:#d97706; --red:#dc2626;
}
.block-container {padding-top:2rem; max-width:1250px;}
/* hero header */
.hero {background:linear-gradient(120deg,#4338ca 0%,#6d28d9 55%,#7c3aed 100%);
  border-radius:18px; padding:26px 30px; color:#fff; margin-bottom:6px;
  box-shadow:0 10px 30px rgba(67,56,202,.25);}
.hero h1 {font-size:30px; font-weight:800; margin:0; letter-spacing:-.5px;}
.hero p {margin:.5rem 0 0; opacity:.92; font-size:15px; max-width:760px; line-height:1.5;}
.hero .pill {display:inline-block; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.3);
  padding:3px 11px; border-radius:999px; font-size:12px; font-weight:600; margin-top:14px; margin-right:6px;}
/* kpi cards */
.kpi {background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px 18px;
  box-shadow:0 1px 2px rgba(15,23,42,.04);}
.kpi .label {color:var(--muted); font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.04em;}
.kpi .value {font-size:28px; font-weight:800; color:var(--ink); line-height:1.1; margin-top:4px;}
.kpi .sub {color:var(--muted); font-size:12px; margin-top:3px;}
/* section titles */
.sec {font-size:13px; font-weight:700; color:var(--brand); text-transform:uppercase;
  letter-spacing:.06em; margin:6px 0 2px;}
/* result card */
.card {background:#fff; border:1px solid var(--line); border-radius:16px; padding:20px 22px;
  box-shadow:0 4px 18px rgba(15,23,42,.06);}
.badge {display:inline-flex; align-items:center; gap:8px; font-weight:800; font-size:18px;
  padding:8px 16px; border-radius:12px; letter-spacing:.2px;}
.badge.green {background:#dcfce7; color:#166534;}
.badge.amber {background:#fef3c7; color:#92400e;}
.badge.red   {background:#fee2e2; color:#991b1b;}
.conf-wrap {background:var(--bg-soft); border-radius:10px; height:10px; margin-top:6px; overflow:hidden;}
.conf-bar  {height:100%; border-radius:10px;}
.rat {background:var(--bg-soft); border-left:3px solid var(--brand); padding:10px 14px;
  border-radius:8px; color:var(--ink); font-size:14px; margin-top:12px;}
.gov {border:1px dashed var(--line); border-radius:10px; padding:12px 14px; margin-top:12px; font-size:14px;}
.step {display:flex; gap:10px; align-items:flex-start; margin-bottom:6px;}
.stepnum {background:var(--brand); color:#fff; border-radius:50%; min-width:22px; height:22px;
  display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)

TRIAGE = {
    "AUTO_APPROVE": {"cls": "green", "icon": "✅", "bar": "#16a34a"},
    "ROUTE_TO_REVIEW": {"cls": "amber", "icon": "⚠️", "bar": "#d97706"},
    "HIGH_PRIORITY_INVESTIGATE": {"cls": "red", "icon": "🚨", "bar": "#dc2626"},
}


@st.cache_data
def _load_and_detect(contamination):
    df = load_claims()
    return detect(df, contamination=contamination)


def kpi(col, label, value, sub=""):
    col.markdown(
        f"<div class='kpi'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


if "audit" not in st.session_state:
    st.session_state.audit = AuditLog()

# ---------------------------------------------------------------- header ------
st.markdown(
    """
<div class="hero">
  <h1>🛡️ Agentic FWA Triage</h1>
  <p>A hybrid machine-learning + agentic-AI prototype for healthcare <b>payment integrity</b>.
  An IsolationForest flags anomalous claims, an autonomous agent investigates and triages each one,
  and a human-in-the-loop governance layer keeps people in control of every decision.</p>
  <span class="pill">Anomaly Detection</span><span class="pill">Agentic AI</span>
  <span class="pill">Human-in-the-Loop Governance</span><span class="pill">Treatment · Payment · Operations</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar -----
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    contamination = st.slider(
        "Detection sensitivity",
        0.05,
        0.30,
        0.15,
        0.01,
        help="Expected share of anomalous claims (IsolationForest contamination). "
        "Higher flags more claims.",
    )
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    mode = st.radio(
        "Agent mode",
        ["Offline (deterministic)", "Live (Anthropic API)"],
        index=1 if has_key else 0,
        captions=["Free, always works", "Real agent reasoning"],
    )
    live = mode.startswith("Live")
    st.caption(
        ("🟢 API key detected" if has_key else "⚪ No API key — offline available")
    )
    if live and not has_key:
        st.warning(
            "Set ANTHROPIC_API_KEY (or a .env file) for live mode. "
            "Falls back to offline automatically if the call fails."
        )
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        "①&nbsp; **Detect** — IsolationForest\n\n"
        "②&nbsp; **Investigate** — agent + tools\n\n"
        "③&nbsp; **Govern** — human-in-the-loop + audit"
    )

# ---------------------------------------------------------------- data + kpis -
scored = _load_and_detect(contamination)
flagged = scored[scored["flagged"]].copy()

truth_anom = scored["_label"] != "normal"
caught = int((scored["flagged"] & truth_anom).sum())
total_anom = int(truth_anom.sum())
false_alarms = int((scored["flagged"] & ~truth_anom).sum())
recall = caught / total_anom if total_anom else 0
fp_rate = false_alarms / len(flagged) if len(flagged) else 0

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Claims processed", f"{len(scored)}", "synthetic dataset")
kpi(k2, "Flagged by ML", f"{len(flagged)}", "sent to the agent")
kpi(
    k3,
    "Detection recall",
    f"{recall:.0%}",
    f"{caught} of {total_anom} anomalies caught",
)
kpi(k4, "False-positive rate", f"{fp_rate:.0%}", "← what the agent triages down")

st.write("")

# ---------------------------------------------------------------- main panes --
left, right = st.columns([1.15, 1])

with left:
    st.markdown(
        "<div class='sec'>Stage 1 — Flagged claims</div>", unsafe_allow_html=True
    )
    st.caption(
        "The detector surfaces statistical outliers. The agent decides which are real."
    )
    show_cols = [
        "claim_id",
        "provider_id",
        "procedure_code",
        "diagnosis_code",
        "units",
        "billed_amount",
        "anomaly_score",
    ]
    st.dataframe(
        flagged[show_cols]
        .style.format({"anomaly_score": "{:.2f}", "billed_amount": "${:,.2f}"})
        .background_gradient(subset=["anomaly_score"], cmap="Reds"),
        height=340,
        use_container_width=True,
        hide_index=True,
    )
    selected = st.selectbox(
        "Select a flagged claim to investigate", flagged["claim_id"].tolist()
    )
    go = st.button("🔎  Investigate claim", type="primary", use_container_width=True)

with right:
    st.markdown(
        "<div class='sec'>Stage 2 &amp; 3 — Investigation &amp; governance</div>",
        unsafe_allow_html=True,
    )
    if go:
        with st.spinner("Agent investigating…"):
            decision = investigate(int(selected), live=live)
            routing = apply_governance(decision)
            st.session_state.audit.record(int(selected), decision, routing)

        t = TRIAGE.get(
            decision["triage"], {"cls": "amber", "icon": "•", "bar": "#64748b"}
        )
        conf = float(decision["confidence"])
        notes = "".join(f"<div>– {n}</div>" for n in routing["notes"])
        human = " · 🧑‍⚖️ human review required" if routing["human_required"] else ""

        st.markdown(
            f"""
        <div class="card">
          <span class="badge {t['cls']}">{t['icon']} {decision['triage'].replace('_',' ')}</span>
          <div style="margin-top:16px; color:var(--muted); font-size:12px; font-weight:600;
               text-transform:uppercase; letter-spacing:.04em;">Agent confidence — {conf:.0%}</div>
          <div class="conf-wrap"><div class="conf-bar" style="width:{conf*100:.0f}%;
               background:{t['bar']};"></div></div>
          <div class="rat"><b>Why:</b> {decision['rationale']}</div>
          <div class="gov"><b>Governance outcome</b><br>{routing['final_routing']}{human}
            <div style="color:var(--muted); font-size:13px; margin-top:6px;">{notes}</div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander(
            f"🧠 Agent reasoning trace — {len(decision['trace'])} tool calls"
        ):
            for i, step in enumerate(decision["trace"], 1):
                st.markdown(
                    f"<div class='step'><div class='stepnum'>{i}</div>"
                    f"<div><code>{step['tool']}</code> &nbsp;<span style='color:#64748b'>"
                    f"{step['input']}</span></div></div>",
                    unsafe_allow_html=True,
                )
                st.json(step["result"], expanded=False)
    else:
        st.info(
            "👈 Pick a flagged claim and click **Investigate** to watch the agent work."
        )

# ---------------------------------------------------------------- audit log ---
st.write("")
st.markdown(
    "<div class='sec'>Audit log — full traceability</div>", unsafe_allow_html=True
)
audit = st.session_state.audit.to_list()
if audit:
    adf = pd.DataFrame(audit)
    st.dataframe(adf, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️  Download audit log (CSV)",
        adf.to_csv(index=False),
        "audit_log.csv",
        "text/csv",
    )
else:
    st.caption(
        "Every investigated claim is logged here — agent decision, confidence, "
        "governance routing, and the tools used — and can be exported for review."
    )
