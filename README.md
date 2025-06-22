# TrustGuard+

A lightweight, end‑to‑end **AI‑powered trust‑and‑safety toolkit** for online marketplaces.  
Given an existing product listing – or a new one you plan to upload – **TrustGuard+** scores how trustworthy it looks and surfaces the exact red‑flags (fake reviews, brand mismatch, suspicious images, abnormal return rates, …).

<div align="center">

```mermaid
flowchart TD
    CSV[Seller / Marketplace<br>CSV export] -->|ingest| INGEST[Ingestion<br> & normalisation]
    subgraph LLM‑Based Analysis
        REVIEWS[Review LLM<br>(Gemini / Flan‑T5)] --> TXT[Text‑fraud score]
        RULES[Business rules<br>(ratings, returns)] --> RULESCORE[Anomaly score]
    end
    subgraph Vision Analysis
        CLIP[CLIP similarity] --> CLIPR[Clip risk]
        BLIP[BLIP‑2 VQA] --> BLIPR[Blip risk]
        CLIPR --> VISCALC[Visual risk combiner]
        BLIPR --> VISCALC
        PADDLE[PaddleOCR] --> OCRTXT[OCR text]
        BRAND[Brand extractor<br>(Flan‑T5)] --> BMFLAG[Brand mismatch?]
        OCRTXT --> BRAND
    end
    INGEST --> REVIEWS
    INGEST --> RULES
    INGEST --> CLIP
    INGEST --> BLIP
    INGEST --> PADDLE
    VISCALC --> AGG[Weighted aggregator]
    TXT --> AGG
    RULESCORE --> AGG
    BMFLAG --> AGG
    AGG --> REPORT[JSON report<br>+ Streamlit dashboard]
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
