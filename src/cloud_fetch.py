"""Cloud step 1: list recordings from the backend and download the ones not yet labelled.

Saves the full list to outputs\\cloud_recordings.csv and files to work\\downloads\\<recording_id>.<ext>
Run:  python src\\cloud_fetch.py            (all unlabelled recordings)
      python src\\cloud_fetch.py --limit 2  (only the first 2 - good for a first test)
"""
import argparse
import csv

import config  # noqa: F401
from config import AUDIO_EXTS, CLOUD_RECORDINGS_CSV, DOWNLOADS_DIR, SENT_LABELS_CSV, VIDEO_EXTS

import cloud

FIELDS = ["recording_id", "filename", "modality", "file_format", "file_size_bytes",
          "duration_seconds", "uploaded_at"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="download at most N new recordings")
    args = ap.parse_args()

    ok, info = cloud.health()
    if not ok:
        raise SystemExit(f"Backend not healthy: {info}")
    recs = cloud.list_recordings()
    with open(CLOUD_RECORDINGS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)

    sent = set()
    if SENT_LABELS_CSV.exists():
        with open(SENT_LABELS_CSV, newline="", encoding="utf-8") as f:
            sent = {row["clip_id"] for row in csv.DictReader(f)}

    supported = {e.lstrip(".") for e in AUDIO_EXTS | VIDEO_EXTS}
    todo = [r for r in recs if r["recording_id"] not in sent]
    skipped_fmt = [r for r in todo if r["file_format"].lower() not in supported]
    todo = [r for r in todo if r["file_format"].lower() in supported]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(recs)} recordings in backend, {len(sent)} already labelled, downloading {len(todo)}")
    for r in skipped_fmt:
        print(f"skip  {r['recording_id']} - unsupported format .{r['file_format']}")

    for i, r in enumerate(todo, 1):
        size_mb = int(r["file_size_bytes"]) / 1e6 if r.get("file_size_bytes") else 0
        path = cloud.download(r, DOWNLOADS_DIR)
        print(f"ok    [{i}/{len(todo)}] {r['recording_id']} ({r['modality']}, {size_mb:.1f} MB) -> {path.name}")
    print("\nCloud fetch done.")


if __name__ == "__main__":
    main()
