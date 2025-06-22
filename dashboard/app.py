import streamlit as st
st.set_page_config(page_title="TrustGuard+ Dashboard", layout="wide")

import json
import pathlib
import pandas as pd

# Base directory of the project
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Sidebar: allow user to override the JSON report path
default_json = ROOT / "my_report.json"
json_path = st.sidebar.text_input("Path to results JSON", str(default_json))
REPORTS = pathlib.Path(json_path)

if not REPORTS.exists():
    st.warning(
        f"Could not find {REPORTS}.\n\n"
        "Run:\n"
        "`python scripts/batch_run.py --csv data/your.csv --out my_report.json`\n"
        "then paste the path above."
    )
    st.stop()

# Read the JSON with replacement for any invalid bytes
json_str = REPORTS.read_text(encoding="utf-8", errors="replace")
data     = json.loads(json_str)

# Build DataFrame and sort by risk (lowest trust first)
df = pd.DataFrame(data).sort_values("trust_score")

# Determine which key columns are present
key_col = "asin" if "asin" in df.columns else "id"
url_col = "product_url" if "product_url" in df.columns else "url"

st.title("TrustGuard+ • Moderator Queue")

st.subheader("All Listings")
st.dataframe(
    df[[key_col, "title", "trust_score", "verdict"]],
    height=300,
    use_container_width=True
)

# Select a single listing to inspect
sel = st.selectbox(
    "Select a listing to inspect",
    df.index,
    format_func=lambda i: f"{df.at[i, key_col]} — {df.at[i,'title']} (score {df.at[i,'trust_score']})"
)

record = df.loc[sel]

st.markdown("## Listing Details")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"**{key_col.upper()}:** {record[key_col]}")
    st.markdown(f"**Title:** {record['title']}")
    st.markdown(f"**URL:** [{record[url_col]}]({record[url_col]})")

with col2:
    st.metric("Trust Score", record["trust_score"])
    st.metric("Verdict", record["verdict"])

st.markdown("### Breakdown of Signals")
if "breakdown" in record:
    st.json(record["breakdown"])

st.markdown("### Explanation")
if "explanation" in record:
    st.json(record["explanation"])
