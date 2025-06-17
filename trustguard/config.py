import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "gemini-1.5-flash"
CLIP_VARIANT   = "ViT-B/32"
KNOWN_BRANDS   = {"nike", "adidas", "puma", "reebok", "converse"}