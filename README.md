# TrustGuard+

_Prototype toolkit for detecting counterfeit or low‑trust product listings on e‑commerce marketplaces (Amazon focus)._

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-Prototype-orange)

---

## ✨ Key Features
| Module | Purpose |
|--------|---------|
| `scripts/batch_run.py` | One‑shot scan of a CSV export of listings. Generates `reports.json`. |
| Dashboard (`dashboard/app.py`) | Streamlit UI for moderators to triage flagged listings. |
| `trustguard/ingest.py` | Robust CSV → internal schema loader (handles messy headers, pipes, JSON blobs). |
| `review_llm.py` | Gemini‑based LLM scoring of review fakery. |
| `visual_clip.py` | Hybrid CLIP + BLIP‑2 image/title consistency check. |
| `brand_match.py` | PaddleOCR + Flan‑T5 brand‑on‑image vs. title mismatch detection. |
| `embed_store.py` | Local FAISS vector store for review similarity search (SBERT). |
| `rules.py` | Heuristic anomaly score (rating/return ratio, etc.). |
| `scoring.py` | Weighted aggregation → 0‑100 Trust Score & **listable / needs review** verdict. |

---

## ⚙️ Architecture

```
CSV → ingest ─┐
              ├─→ review_llm                    ├─→ visual_clip      \            +→ `reports.json`
              ├─→ brand_match       > aggregate |
              └─→ rules            /            +→ dashboard
```

* **LLM:** Gemini 1.5 flash (free‑tier safe‑prompt)  
* **Image models:** OpenAI CLIP (ViT‑B/32) + Salesforce BLIP‑2 OPT‑2.7B  
* **OCR:** PaddleOCR (English) → Flan‑T5 brand extractor  
* **Embeddings:** `all‑MiniLM‑L6‑v2` (384‑d) with FAISS inner‑product index

---

## 🏗 Installation

```bash
git clone https://github.com/your‑org/trustguard.git
cd trustguard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Extra system deps**

| Component | Requirement |
|-----------|-------------|
| PaddleOCR | paddlespeech‑friendly wheels (`pip install paddleocr`) <br> MSVC ++ build tools on Windows |
| CLIP / BLIP‑2 | CUDA 11+ if you want GPU acceleration (else runs on CPU) |
| Tesseract | _Not used_ (PaddleOCR replaces it). |

Set your Gemini API key:

```bash
export GOOGLE_API_KEY="your‑api‑key"
```

---

## 🚀 Quick Start

### 1. Run a batch scan

```bash
python scripts/batch_run.py --csv data/amazon_sneakers_all2.csv --out my_report.json
```

You’ll get a per‑listing object like:

```json
{
  "asin": "B0D799T6K4",
  "trust_score": 13,
  "verdict": false,
  "breakdown": {
    "text_score": 0.95,
    "visual_score": 0.76,
    "brand_mismatch": false,
    "rule_score": 0.90
  },
  "explanation": { "text": "Identical reviews suggest coordinated fake reviews." }
}
```

### 2. Open the moderator dashboard

```bash
streamlit run dashboard/app.py -- --report my_report.json
```

---

## 🧮 Scoring Formula

```
risk_text    = text_score          # 0‑1 from LLM
risk_visual  = visual_score        # 0‑1 hybrid CLIP/BLIP
risk_rule    = rule_score          # 0‑1 heuristics
risk_brand   = 1.0 if brand_mismatch else 0.0

trust = 1 - (0.20*risk_text + 0.20*risk_visual + 0.20*risk_rule + 0.40*risk_brand)
trust_score = int(trust*100)        # 0‑100
verdict = trust_score >= 70         # listable?
```

Weights live in [`trustguard/scoring.py`](trustguard/scoring.py). Adjust them to suit your policy.

---

## 📁 Repo Layout

```
.
├─ trustguard/            # Core library
│   ├─ ingest.py
│   ├─ review_llm.py
│   ├─ visual_clip.py
│   ├─ brand_match.py
│   ├─ embed_store.py
│   ├─ rules.py
│   └─ scoring.py
├─ scripts/
│   └─ batch_run.py
├─ dashboard/
│   └─ app.py
└─ requirements.txt
```

---

## ✍️ Extending

* Plug in a different LLM by swapping `review_llm._query_llm`.
* Change image risk model: e.g., swap BLIP‑2 for LLaVA.
* Add new rule heuristics in `rules.py` — the aggregate function will pick them up.
* Increase OCR languages: `PaddleOCR(lang="en+hi")` etc.

---

## 🪪 License

Released under the **MIT License** — free to use, modify and redistribute with attribution.

---

> _Prototype built for HackOn with Amazon – Season 5 (Trust & Safety theme)._  
> Maintainer: **Harish M.** • Contributions welcome!
