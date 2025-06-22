# TrustGuard+

TrustGuard+ is a prototype for automated trust scoring of Amazon product listings, identifying potential counterfeits or fraudulent reviews.

## Features

- **Ingestion**: CSV loader grouping by ASIN, aggregates images, reviews, ratings.
- **Review Fraud Detection**: Uses an LLM to score review authenticity.
- **Visual Verification**: Combines CLIP and a vision–LLM model (e.g., BLIP-2) to compare titles against images.
- **Brand Matching**: OCR via PaddleOCR and a Flan-T5-based extractor to verify brand names.
- **Anomaly Detection**: Heuristics on ratings and returns to flag suspicious patterns.
- **Scoring & Verdict**: Weighted aggregation into a trust score (0–100) and listable flag.
- **Dashboard**: Streamlit apps for moderators and sellers.

## Installation

### Requirements

- Python 3.8+
- `torch`, `transformers`, `clip`, `sentence-transformers`, `faiss-cpu`
- `paddleocr`, `google-generativeai`
- `streamlit`

### Setup

```bash
pip install -r requirements.txt
```

## Usage

### Batch Scoring

```bash
python scripts/batch_run.py --csv data/amazon_sneakers_all2.csv --out reports.json
```

### Moderator Dashboard

```bash
streamlit run dashboard/app.py
```

### Seller Dashboard (Single-Product Upload)

```bash
streamlit run dashboard/seller_app.py
```

## Architecture

The pipeline consists of:

1. **Ingestion** (`ingest.py`): Reads CSV, groups by ASIN, aggregates metadata.
2. **Orchestration** (`orchestrator.py`): Invokes each signal module and aggregates results.
3. **Embedding Store** (`embed_store.py`): Caches and indexes review embeddings with FAISS.
4. **Review LLM** (`review_llm.py`): Uses an LLM to detect fake reviews.
5. **Visual Scoring** (`visual_clip.py`): Computes image–title similarity via CLIP and BLIP-2.
6. **Brand Matching** (`brand_match.py`): Extracts and compares brand text via OCR and LLM.
7. **Rules** (`rules.py`): Applies simple heuristics on ratings/returns.
8. **Scoring** (`scoring.py`): Aggregates all signal scores.

## Configuration

Edit `trustguard/config.py` with your credentials and model settings:

```python
GOOGLE_API_KEY = "your-google-api-key"
CLIP_VARIANT = "ViT-B/32"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-pro"
GEMINI_VISION_MODEL = "gemini-2.5-flash"
```

## License

MIT
