"""Prueba de carga asíncrona: latencia p50/p95/p99, throughput y tasa de error por
nivel de concurrencia; el punto de saturación es el primer nivel donde el
throughput deja de crecer > 10 % o la p95 se duplica. (Alternativa: `locust -f
scripts/locustfile.py`.)"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

TEXTS = ["python sql pipeline deploy cloud", "customer sales negotiation target", "figma prototype typography layout"]


async def run(url: str, conc: int, n: int, endpoint: str, img: bytes | None) -> dict:
    lat, errs = [], 0
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=30) as c:
        async def one(i: int) -> None:
            nonlocal errs
            async with sem:
                t = time.perf_counter()
                try:
                    if endpoint == "text":
                        r = await c.post(f"{url}/predict_text", json={"text": TEXTS[i % 3]})
                    else:
                        r = await c.post(f"{url}/predict_image", files={"file": ("x.png", img, "image/png")})
                    errs += r.status_code != 200
                except httpx.HTTPError:
                    errs += 1
                lat.append((time.perf_counter() - t) * 1e3)
        t0 = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(n)])
        dt = time.perf_counter() - t0
    a = np.array(lat)
    return dict(endpoint=endpoint, concurrency=conc, p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), throughput=n / dt, error_rate=errs / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--requests", type=int, default=400)
    a = ap.parse_args()
    import sys

    sys.path.insert(0, "src")
    import cv2

    from project.data_seg import make_synthetic_image
    img = cv2.imencode(".png", make_synthetic_image((520, 704), "cort", seed=0)[0])[1].tobytes()
    rows = []
    for ep in ("text", "image"):
        for c in a.concurrency:
            r = asyncio.run(run(a.url, c, a.requests if ep == "text" else a.requests // 4, ep, img)); rows.append(r); print(r)
        ep_rows = [r for r in rows if r["endpoint"] == ep]
        sat = next((ep_rows[i]["concurrency"] for i in range(1, len(ep_rows))
                    if ep_rows[i]["throughput"] < 1.1 * ep_rows[i - 1]["throughput"] or ep_rows[i]["p95"] > 2 * ep_rows[0]["p95"]), None)
        print(f"[{ep}] punto de saturación ≈ concurrencia {sat}")
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/tables/load_test.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
