# TrustGuard +

<div align="center">
  <img alt="TrustGuard+ logo" src="https://github.com/Mharish0341/trustguard/assets/logo.png" height="95"/>
  <br/>
  <b>Instant counterfeit & trust-fraud detection for e-commerce listings</b>
  <br/><br/>
</div>

---

## ✨ Overview
**TrustGuard +** is a drop-in Python toolkit that audits any catalogue export (CSV / API feed) and assigns a transparent **_Trust Score (0–100)_** for every product.  
It inspects **reviews, images, ratings, returns & brand logos** using lightweight open-source models that run locally or behind your own firewall. Moderators get a JSON report and a Streamlit dashboard – no cloud vendor lock-in.

| Module | Models under the hood | Flags                         |
|--------|----------------------|-------------------------------|
| **Review LLM**       | Gemini 1.5 Pro / Flan-T5    | Copy-pasted or bot-generated reviews |
| **Vision Risk**      | CLIP + BLIP-2 (OPT 2.7 B)  | Stock / unrelated product photos     |
| **Brand Match**      | PaddleOCR + Flan-T5        | Logo on image ≠ brand in title        |
| **Rules Engine**     | Simple heuristics          | Rating spikes, abnormal return ratios |

Everything is fused through a weighted aggregator and explained in plain English.

---

```bash
# 1.  Clone & install
git clone https://github.com/your-org/trustguard.git
cd trustguard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # CPU-only by default

# 2.  Run a batch audit
python scripts/batch_run.py \
      --csv data/amazon_sneakers_all2.csv \
      --out reports.json

# 3.  Browse results
streamlit run dashboard/app.py


```mermaid
flowchart TD
    %% ────────────────────────── Ingestion ──────────────────────────
    CSV["CSV export<br/>Seller / Marketplace"] --> INGEST[Ingestion<br/>+ normalisation]

    %% ──────────────────────── LLM-based analysis ───────────────────
    subgraph "LLM-Based Analysis"
        REVIEWS["Review LLM<br/>(Gemini / Flan-T5)"] --> TXT["Text-fraud score"]
        RULES["Business rules<br/>(ratings + returns)"] --> RULESCORE["Anomaly score"]
    end

    %% ───────────────────────── Vision analysis ─────────────────────
    subgraph "Vision Analysis"
        CLIP["CLIP similarity"] --> CLIPR["CLIP risk"]
        BLIP["BLIP-2 VQA"] --> BLIPR["BLIP risk"]
        PADDLE["PaddleOCR"] --> OCRTXT["OCR text"]
        OCRTXT --> BRAND["Brand extractor<br/>(Flan-T5)"]
        BRAND --> BMFLAG["Brand mismatch?"]
        CLIPR --> VISCALC["Visual-risk combiner"]
        BLIPR --> VISCALC
    end

    %% ─────────────── Data flow from ingest to individual blocks ────
    INGEST --> REVIEWS
    INGEST --> RULES
    INGEST --> CLIP
    INGEST --> BLIP
    INGEST --> PADDLE

    %% ───────────────────────── Aggregation ─────────────────────────
    VISCALC --> AGG["Weighted aggregator"]
    TXT --> AGG
    RULESCORE --> AGG
    BMFLAG --> AGG

    %% ─────────────────────────── Output ────────────────────────────
    AGG --> REPORT["JSON report<br/>Streamlit dashboard"]

```

</div>

---

## Key Modules

| Folder | What it does |
|--------|--------------|
| `trustguard/ingest.py` | Normalises any Amazon–style CSV into a clean, deduped stream of listings. |
| `trustguard/review_llm.py` | Uses a cached LLM (Gemini 1.5 Pro or Flan‑T5) to spot review fraud. |
| `trustguard/visual_clip.py` | Combines **CLIP** similarity with **BLIP‑2** VQA for image/title consistency. |
| `trustguard/brand_match.py` | Runs **PaddleOCR** → extracts brand with Flan‑T5 → fuzzy‑matches title. |
| `trustguard/rules.py` | Simple statistical rules (rating distribution, return spikes). |
| `trustguard/scoring.py` | Final weighted aggregation → _Trust Score_ (0‑100) & verdict. |
| `scripts/batch_run.py` | One‑shot CSV → `reports.json`. |
| `dashboard/app.py` | Streamlit moderator queue. |

---

## Quick start

```bash
# 1. Install deps (CUDA optional)
pip install -r requirements.txt

# 2. Run a batch audit
python scripts/batch_run.py --csv data/amazon_sneakers_all2.csv --out reports.json

# 3. Open the dashboard
streamlit run dashboard/app.py
```

Environment variables required:

| Name | Purpose |
|------|---------|
| `GOOGLE_API_KEY` | Gemini review‑fraud scoring & OCR fallback |
| `LLM_MODEL` | e.g. `gemini-1.5-pro` or `google/flan-t5-large` |
| `CLIP_VARIANT` | OpenAI `ViT‑L/14@336px` works well |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |

---

## Scoring rubric

| Signal | Weight | Meaning of **1.0** |
|--------|--------|--------------------|
| `text_score` | 0.20 | reviews look entirely fake |
| `visual_score` | 0.20 | title/image clearly unrelated |
| `rule_score` | 0.20 | extreme rating / return anomaly |
| `brand_mismatch` | 0.40 | brand on image **≠** brand in title |

`Trust Score = 100 × (1 – weighted risk)`  
Default _listable_ threshold = **70**.

---

## Contributors

* **Harish Muthubalakrishnan**  
* **Benicia A**  
* **Mahalakshmi P C**

---

## Disclaimer

Prototype only – not affiliated with Amazon.  
All brand names & images belong to their respective owners.
