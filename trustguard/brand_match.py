from typing import List
import re
import numpy as np
from PIL import Image

import easyocr
from .utils import fetch_image_bytes, bytes_to_pil
from .config import KNOWN_BRANDS

# Load once at import
# languages=["en"] can be extended if you expect reviews/logos in other scripts
_reader = easyocr.Reader(["en"], gpu=False)  


def brand_mismatch(image_url: str, title: str) -> bool:
    """
    Downloads the image, runs OCR, and returns True if any expected
    brand (found in the title) is NOT detected in the image text.
    """
    # 1) Download & open
    try:
        pil_img = bytes_to_pil(fetch_image_bytes(image_url))
    except Exception:
        # If download fails, skip OCR check
        return False

    # 2) Run EasyOCR (returns list of strings)
    try:
        ocr_results: List[str] = _reader.readtext(
            np.array(pil_img), detail=0, paragraph=True
        )
    except Exception as e:
        print(f"[brand_match] EasyOCR error: {e}; skipping check.")
        return False

    full_text = " ".join(ocr_results).lower()

    # 3) Check for each brand keyword from the title
    expected_brands = [b for b in KNOWN_BRANDS if b in title.lower()]
    for brand in expected_brands:
        norm = re.sub(r"[^a-z]", "", brand)
        if norm not in full_text.replace(" ", ""):
            # Brand mentioned in title but not found via OCR
            return True

    # No mismatches found
    return False
