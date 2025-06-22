from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, Generator, List

import pandas as pd

_BIDI = r"\u200e\u200f\u202a-\u202e"

def _clean_header(h: str) -> str:
    h = unicodedata.normalize("NFKC", h)
    h = re.sub(f"[{_BIDI}]", "", h)
    h = re.sub(r"\s+", " ", h).strip().lower().replace(" ", "_")
    return h

def _split_pipe(cell: str) -> List[str]:
    return [p.strip() for p in str(cell).split("|") if p.strip()]

def _collect_images(row) -> List[str]:
    for key in ("images", "image_urls", "img_url"):
        if key in row and row[key]:
            return _split_pipe(row[key])
    return []

def load_listings(csv_path: Path) -> Generator[Dict[str, Any], None, None]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = [_clean_header(c) for c in df.columns]

    # find ASIN column
    asin_col = next((c for c in df.columns if re.fullmatch(r"asin_*\b", c)), None)
    if not asin_col:
        raise ValueError("CSV must contain an ASIN column (case-insensitive)")

    for asin, grp in df.groupby(asin_col):
        first = grp.iloc[0]
        img_set, ratings, returns, review_bodies = set(), [], 0, []

        # parse returns/return_count once
        for rc in ("return_count", "returns"):
            if first.get(rc):
                try:
                    returns = int(first[rc])
                except:
                    returns = 0
                break

        for _, row in grp.iterrows():
            img_set.update(_collect_images(row))

            if row.get("rating"):
                try:
                    ratings.append(float(row["rating"]))
                except:
                    pass

            if row.get("reviews_json"):
                try:
                    arr = json.loads(row["reviews_json"])
                    for o in arr:
                        body = o.get("body") or o.get("text") or ""
                        if body:
                            review_bodies.append(body.strip())
                    continue
                except:
                    pass

            if row.get("review_texts"):
                for part in str(row["review_texts"]).split("||"):
                    b = part.strip()
                    if b:
                        review_bodies.append(b)

        yield {
            "id":          asin,
            "url":         first.get("product_url") or f"https://www.amazon.in/dp/{asin}",
            "title":       first.get("title", ""),
            "description": first.get("description", ""),
            "images":      list(img_set),
            "reviews":     review_bodies,
            "ratings":     ratings,
            "returns":     returns,
        }
