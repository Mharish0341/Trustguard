TrustGuard+ • Marketplace Trust Analyst
Moderators & Sellers Dashboard for Amazon Listings
TrustGuard+ analyzes Amazon product listings to detect potential fraud, counterfeit risk, and anomalous behavior—helping moderators flag suspicious products and informing sellers whether it’s safe to list their items.

Key Objectives
Automated Trust Scoring
Combine textual, visual, brand, and behavioral signals into a single 0–100 trust score.

Explainable Verdicts
Provide a clear “Pass/Flag” (or “Listable/Not Listable”) decision along with a signal breakdown and concise rationale.

Dual Path: Moderator & Seller

Moderator view: Batch-process a CSV of listings and review a queue of flagged products.

Seller view: Upload a single product to get instant feedback on listing safety.

Features
Textual Analysis
Leverage an LLM (e.g. Flan-T5) to score review authenticity (fake vs. genuine).

Visual Consistency

Compute CLIP similarity between title and images.

BLIP-2 VQA model rates “title→image” match quality for deeper multimodal checks.

Brand Verification

OCR (PaddleOCR) + LLM fallback to extract brand from image.

Fuzzy‐match against title’s brand to spot mislabeling.

Rule‐based Anomalies
Simple heuristics on 5-star review ratios and return counts to catch unusual patterns.

Embedding DB (Optional)
Incrementally build a FAISS index of past review embeddings for similarity‐based detective work.

Configurable Weighting
Fine-tune the relative importance of text, visual, rule, and brand signals (e.g. upweight brand mismatch).

Quickstart
Clone & Install

bash
Copy
Edit
git clone https://github.com/YourOrg/TrustGuardPlus.git
cd TrustGuardPlus
pip install -r requirements.txt
Prepare Listings CSV
Ensure your CSV has an ASIN column and at least one of: images, image_urls, or img_url; plus reviews_json or review_texts.

Run Batch Analysis (Moderator)

bash
Copy
Edit
python scripts/batch_run.py --csv data/your_listings.csv --out reports.json
View Dashboard

bash
Copy
Edit
streamlit run dashboard/app.py
Browse the Moderator Queue or switch to the Seller page to upload a single listing for instant verdict.

Tech Stack
Component	Library / Service
LLM	Google Gemini / Flan-T5-Large
Vision Models	OpenAI CLIP, BLIP-2 (OPT-2.7B)
OCR	PaddleOCR + LLM fallback
Embeddings	Sentence-Transformers, FAISS
Dashboard	Streamlit
Lang & Tools	Python 3.10+, PyTorch, Transformers, NumPy, Pandas, Requests

Architecture
Ingest:
Load CSV → normalize headers → group by ASIN → collect images, ratings, returns, review texts.

Orchestrator:
For each listing:

review_fraud_score → LLM JSON → text_score

weighted_visual_risk → CLIP + BLIP-2 → visual_score

brand_mismatch → PaddleOCR + Flan-T5 → brand_flag

anomaly_score → heuristic on ratings/returns → rule_score

aggregate → weighted sum → trust_score, verdict

Output:
JSON with ASIN, title, URL, trust_score, boolean verdict, signal breakdown, and short explanations.

Signal Weighting
We recommend the following defaults (sum to 1.0):

Signal	Weight
Text Reviews	0.20
Visual Match	0.20
Rule Heuristic	0.20
Brand Match	0.40

python
Copy
Edit
def aggregate(text_s, visual_s, rule_s, brand_flag, threshold=70):
    w_text, w_visual, w_rule, w_brand = 0.20, 0.20, 0.20, 0.40
    risk_brand = 1.0 if brand_flag else 0.0
    trust_frac = (
        (1 - text_s)   * w_text
      + (1 - visual_s) * w_visual
      + (1 - rule_s)   * w_rule
      + (1 - risk_brand)* w_brand
    )
    score = int(trust_frac * 100)
    listable = score >= threshold
    return score, listable
}
Future Scope
Seller-Facing App: Streamlined UI for individual uploads and actionable guidance.

Continuous Learning: Retrain LLMs and vision models on flagged data to improve detection.

Multilingual & Internationalization: Support global marketplaces beyond Amazon India.

Automated Alerts: Slack/email notifications for high-risk listings.
