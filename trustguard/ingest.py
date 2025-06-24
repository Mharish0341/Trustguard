from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, Generator, List, Set

import pandas as pd

_BIDI   = r"\u200e\u200f\u202a-\u202e"          # zero-width bidi chars
_IMG_KEYS = ("image", "img_url", "image_urls", "images")

def _clean(h: str) -> str:
    h = unicodedata.normalize("NFKC", h)
    h = re.sub(f"[{_BIDI}]", "", h)
    h = re.sub(r"\s+", " ", h).strip().lower().replace(" ", "_")
    return h

def _split(cell: str) -> List[str]:
    return [p.strip() for p in str(cell).split("|") if p.strip()]

def _json_reviews(text: str) -> List[str]:
    try:
        data = json.loads(text)
        return [o.get("body", "").strip() for o in data if isinstance(o, dict)]
    except json.JSONDecodeError:
        return []


def load_listings(csv_path: Path) -> Generator[Dict[str, Any], None, None]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = [_clean(c) for c in df.columns]

    asin_col  = next(c for c in df.columns if re.fullmatch(r"asin_*", c))
    type_col  = "type" if "type" in df.columns else None
    url_col   = next((c for c in df.columns if c.startswith("http") or "product_url" in c), None)
    rating_col= "rating" if "rating" in df.columns else None
    returns_col = next((c for c in df.columns if c in ("returns", "return_count")), None)

    for asin, grp in df.groupby(asin_col, sort=False):
        product_row = grp
        if type_col and "product" in grp[type_col].str.lower().values:
            product_row = grp[grp[type_col].str.lower() == "product"].iloc[0]
        else:
            product_row = grp.iloc[0]

        images: Set[str] = set()
        for col in _IMG_KEYS:
            if col in df.columns:
                if type_col:
                    images.update(grp.loc[grp[type_col].str.lower() == "image", col])
                else:
                    images.update(grp[col])
        images = {u for u in images if u.startswith("http")}

        reviews: List[str] = []
        if "reviews_json" in df.columns and product_row["reviews_json"]:
            reviews = _json_reviews(product_row["reviews_json"])
        if not reviews and "review_texts" in df.columns and product_row["review_texts"]:
            for part in _split(product_row["review_texts"]):
                reviews.append(part)

        ratings: List[float] = []
        if rating_col:
            for r in grp[rating_col]:
                try:
                    ratings.append(float(r))
                except ValueError:
                    pass

        returns = 0
        if returns_col and product_row[returns_col]:
            try:
                returns = int(product_row[returns_col])
            except ValueError:
                returns = 0

        yield {
            "id":          asin,
            "url":         product_row.get(url_col) or f"https://www.amazon.in/dp/{asin}",
            "title":       product_row.get("title", ""),
            "description": product_row.get("description", ""),
            "images":      list(images),
            "reviews":     reviews,      # list[str]
            "ratings":     ratings,      # list[float]
            "returns":     returns,      # int
        }
