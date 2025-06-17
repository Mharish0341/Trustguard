# TrustGuard+  🚀

Hybrid Visual‑AI + LLM Trust & Safety engine  
*Gemini Flash 1.5*    *text‑embed‑004*    *FAISS*

```
pip install -r requirements.txt
cp .envexample .env  # add your GOOGLE_API_KEY
python scripts/batch_run.py --csv data/sample_listings.csv
streamlit run dashboard/app.py
```

### Architecture
```
Listing →  Orchestrator  → (Review‑LLM │ Visual‑AI │ Brand‑Match │ Rules)
                                       ↘  Trust‑Score Aggregator  ↘
                                        Moderator Dashboard   Buyer API
```