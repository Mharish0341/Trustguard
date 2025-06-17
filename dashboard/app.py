import json
import pathlib

import pandas as pd
import streamlit as st

# ---------- Paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]   # repo root
REPORTS = ROOT / "reports.json"

# ---------- UI ----------
st.set_page_config(page_title="TrustGuard+ Dashboard", layout="wide")

if not REPORTS.exists():
    st.warning(
        f"Could not find {REPORTS}.  "
        "Run `python scripts/batch_run.py --csv your_file.csv` first."
    )
    st.stop()

data = json.loads(REPORTS.read_text())
df = pd.DataFrame(data).sort_values("trust_score")

st.title("TrustGuard+ • Moderator Queue")
st.dataframe(df[["id", "trust_score", "verdict"]], height=400)

row_idx = st.selectbox(
    "Select listing to inspect",
    df.index,
    format_func=lambda i: f"{df.loc[i,'id']} (score {df.loc[i,'trust_score']})",
)

record = df.loc[row_idx]

# Quick headline metrics
st.metric("Trust Score", record["trust_score"])
st.metric("Verdict", record["verdict"])

st.subheader("Explanation")
st.json(record["explain"])
