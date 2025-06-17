from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Generator, List, Any

import pandas as pd

_BIDI = r"\u200e\u200f\u202a-\u202e"  # zero-width & bidi chars

def _clean_header(h: str) -> str:
    h = unicodedata.normalize("NFKC", h)
    h = re.sub(f"[{_BIDI}]", "", h)        # remove bidi / zero-width
    h = re.sub(r"\s+", " ", h).strip()
    h = h.lower().replace(" ", "_")
    return h

def _split_pipe(cell: str) -> List[str]:
    return [p.strip() for p in str(cell).split("|") if p.strip()]

def _parse_reviews(row) -> List[str]:
    if row.get("reviews_json"):
        try:
            objs = json.loads(row["reviews_json"])
            return [o.get("body", "").strip() for o in objs if o.get("body")]
        except json.JSONDecodeError:
            pass
    if row.get("review_texts"):
        return [t.strip() for t in str(row["review_texts"]).split("||") if t.strip()]
    return []

def _collect_images(row) -> List[str]:
    for key in ("images", "image_urls", "img_url"):
        if key in row and row[key]:
            return _split_pipe(row[key])
    return []


def load_listings(csv_path: Path) -> Generator[Dict[str, Any], None, None]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # normalise headers
    df.columns = [_clean_header(c) for c in df.columns]

    # find ASIN column (allow variants like 'asin\u200f')
    asin_col = next((c for c in df.columns if re.fullmatch(r"asin_*", c)), None)
    if asin_col is None:
        raise ValueError("CSV must contain an ASIN column (case-insensitive)")

    for asin, grp in df.groupby(asin_col):
        first = grp.iloc[0]

        # aggregate
        img_set: set[str] = set()
        reviews: list[str] = []
        ratings: list[str] = []

        for _, row in grp.iterrows():
            img_set.update(_collect_images(row))
            reviews.extend(_parse_reviews(row))
            if "rating" in row and row["rating"]:
                ratings.append(str(row["rating"]))

        yield {
            "id": asin,
            "url": first.get("product_url") or f"https://www.amazon.in/dp/{asin}",
            "title": first.get("title", ""),
            "description": first.get("description", ""),
            "images": list(img_set),
            "reviews": reviews,
            "ratings": ratings,
            "returns": int(first.get("return_count", 0) or 0),
        }
