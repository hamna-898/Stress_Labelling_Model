"""Cloud step 0: read-only connection test. Changes nothing, downloads nothing.
Run:  python src\\cloud_check.py
"""
import csv
from collections import Counter

import config  # noqa: F401
from config import SENT_LABELS_CSV

import cloud

print("API:", cloud.BASE)
ok, info = cloud.health()
print("Health:", info)
if not ok:
    raise SystemExit("FAIL: backend health check did not return 200 - tell the backend owner which part is Disconnected.")

recs = cloud.list_recordings()
print(f"Token OK. {len(recs)} recordings in the backend.")
print("  by modality:", dict(Counter(r["modality"] for r in recs)))
print("  by format  :", dict(Counter(r["file_format"] for r in recs)))
sent = set()
if SENT_LABELS_CSV.exists():
    with open(SENT_LABELS_CSV, newline="", encoding="utf-8") as f:
        sent = {row["clip_id"] for row in csv.DictReader(f)}
print(f"  already labelled by this pipeline: {sum(r['recording_id'] in sent for r in recs)}")
for r in recs[:5]:
    print(f"  - {r['recording_id']}  {r['modality']:5s} {r['file_format']:4s} "
          f"{int(r['file_size_bytes'])/1e6:6.1f} MB  {r['filename']}")
print("\nCloud check PASSED")
