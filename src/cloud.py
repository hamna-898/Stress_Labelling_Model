"""Client for the Silent Cries backend API (the /api/ml/* endpoints).

The ML token is read from D:\\Silent Cries\\.env  (ML_API_TOKEN=...). It never appears in code.
Every call retries on dropped connections / server errors, because downloads can be cut off.
"""
import os
import time
from pathlib import Path

import requests

from config import API_BASE_URL_DEFAULT, ENV_FILE


def _load_env_file():
    """Read KEY=VALUE lines from .env into environment variables (no extra package needed)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
BASE = os.environ.get("API_BASE_URL", API_BASE_URL_DEFAULT).rstrip("/")


class ApiError(RuntimeError):
    pass


def _headers():
    token = os.environ.get("ML_API_TOKEN", "").strip()
    if not token or token.lower().startswith("paste"):
        raise SystemExit(f"ML_API_TOKEN is not set. Put it in {ENV_FILE} as:  ML_API_TOKEN=<token>")
    return {"Authorization": f"Bearer {token}"}


def _error_text(r):
    try:
        return r.json().get("error", r.text[:200])
    except ValueError:
        return r.text[:200]


def _request(method, path, auth=True, retries=5, **kwargs):
    """HTTP call with retries on network errors and 5xx (except 503 'not configured', which won't fix itself)."""
    url = f"{BASE}{path}"
    if auth:
        kwargs["headers"] = {**kwargs.get("headers", {}), **_headers()}
    for attempt in range(1, retries + 1):
        try:
            r = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == retries:
                raise ApiError(f"{method} {path}: network error after {retries} tries ({type(e).__name__})")
            wait = 2 ** attempt
            print(f"   network problem, retrying in {wait}s ({attempt}/{retries})...")
            time.sleep(wait)
            continue
        if r.status_code >= 500 and r.status_code != 503 and attempt < retries:
            time.sleep(2 ** attempt)
            continue
        return r
    return r


def health():
    """GET /api/health -> (ok: bool, details: dict). No token needed."""
    r = _request("GET", "/api/health", auth=False)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:200]}
    return r.status_code == 200, body


def list_recordings(page_size=100):
    """All recordings, oldest first, following the limit/offset paging."""
    offset, out = 0, []
    while True:
        r = _request("GET", "/api/ml/recordings", params={"limit": page_size, "offset": offset})
        if r.status_code != 200:
            raise ApiError(f"list recordings failed: {r.status_code} {_error_text(r)}")
        page = r.json()
        out.extend(page["recordings"])
        if page["count"] < page["limit"]:
            return out
        offset += page["limit"]


def download(rec, dest_dir: Path, retries=4) -> Path:
    """Download one recording to <dest_dir>/<recording_id>.<ext>. Incomplete files are retried."""
    rid, ext = rec["recording_id"], (rec.get("file_format") or "bin").lower()
    dest = dest_dir / f"{rid}.{ext}"
    expected = int(rec["file_size_bytes"]) if rec.get("file_size_bytes") else None
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(f"{BASE}/api/ml/recordings/{rid}/download", headers=_headers(),
                              stream=True, timeout=120) as r:
                if r.status_code != 200:
                    raise ApiError(f"download {rid} failed: {r.status_code} {_error_text(r)}")
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            size = part.stat().st_size
            if expected is not None and size != expected:
                raise ApiError(f"incomplete download ({size} of {expected} bytes)")
            part.replace(dest)
            return dest
        except (requests.ConnectionError, requests.Timeout, ApiError) as e:
            if isinstance(e, ApiError) and "failed: 4" in str(e):
                raise                                   # 400/401/404 won't fix themselves
            if attempt == retries:
                raise ApiError(f"download {rid}: gave up after {retries} tries ({e})")
            print(f"   download problem ({e}); retrying ({attempt}/{retries})...")
            time.sleep(2 ** attempt)


def send_labels(items):
    """POST /api/ml/predictions in batches of <= 500. items: [{recording_id, model_label, model_confidence?}]"""
    saved = []
    for i in range(0, len(items), 500):
        r = _request("POST", "/api/ml/predictions", json={"predictions": items[i:i + 500]})
        if r.status_code != 200:
            raise ApiError(f"saving labels failed: {r.status_code} {_error_text(r)}")
        saved.extend(r.json()["predictions"])
    return saved
