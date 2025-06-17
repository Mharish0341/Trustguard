from functools import lru_cache
import requests, io, asyncio, aiohttp
from PIL import Image

@lru_cache(maxsize=1024)
def fetch_image_bytes(url: str) -> bytes:
    return requests.get(url, timeout=10).content

async def fetch_many(urls):
    async with aiohttp.ClientSession() as sess:
        async def grab(u):
            async with sess.get(u, timeout=15) as resp:
                return await resp.read()
        tasks = [grab(u) for u in urls]
        return await asyncio.gather(*tasks)

def bytes_to_pil(b: bytes):
    return Image.open(io.BytesIO(b)).convert("RGB")