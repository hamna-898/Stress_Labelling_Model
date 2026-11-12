"""Cloud step 3: send final labels to the backend, then delete the raw media for those recordings.

Sends a label when the recording has no label from us yet, or when our label/fusion version changed.
Clips still waiting for a model step are not sent. Keeps features/embeddings/CSVs; deletes only
the downloaded file, its extracted .wav and its frames (DELETE_MEDIA_AFTER_SEND in config.py).
Run:  python src\\cloud_send.py
"""
import csv
import shutil
from datetime import datetime, timezone

import config  # noqa: F401
from config import (AUDIO_DIR, CLOUD_RECORDINGS_CSV, DELETE_MEDIA_AFTER_SEND, DOWNLOADS_DIR,
                    FRAMES_DIR, FUSED_CSV, SEND_CONFIDENCE, SENT_LABELS_CSV)

import pandas as pd

import cloud

SENT_FIELDS = ["clip_id", "model_label", "confidence", "fused_score", "fusion_version", "sent_at"]


def cleanup(rid):
    for p in DOWNLOADS_DIR.glob(f"{rid}.*"):
        p.unlink(missing_ok=True)
    (AUDIO_DIR / f"{rid}.wav").unlink(missing_ok=True)
    shutil.rmtree(FRAMES_DIR / rid, ignore_errors=True)


def main():
    if not FUSED_CSV.exists() or not CLOUD_RECORDINGS_CSV.exists():
        raise SystemExit("Run cloud_fetch.py, the model steps and fuse.py first.")
    fused = pd.read_csv(FUSED_CSV, dtype={"clip_id": str})
    cloud_ids = set(pd.read_csv(CLOUD_RECORDINGS_CSV, dtype=str)["recording_id"])
    sent = pd.read_csv(SENT_LABELS_CSV, dtype=str) if SENT_LABELS_CSV.exists() else pd.DataFrame(columns=SENT_FIELDS)
    if SENT_LABELS_CSV.exists() and list(sent.columns) != SENT_FIELDS:        # upgrade an older file's columns
        sent = sent.reindex(columns=SENT_FIELDS)
        sent.to_csv(SENT_LABELS_CSV, index=False)
    sent = sent.fillna("")
    # what we last sent per recording: (label, fusion version, whether a confidence was included)
    last_sent = {r.clip_id: (r.model_label, r.fusion_version, r.confidence != "") for r in sent.itertuples()}

    fused = fused[fused["clip_id"].isin(cloud_ids)]
    waiting = fused[fused["reasons"].fillna("").str.contains("not run yet")]
    ready = fused.drop(waiting.index)
    items, rows = [], []
    for r in ready.itertuples():
        label = "Stressed" if int(r.fused_label) == 1 else "Not Stressed"
        confidence = round(float(r.fused_score) if label == "Stressed" else 1 - float(r.fused_score), 4)
        if last_sent.get(r.clip_id) == (label, r.fusion_version, SEND_CONFIDENCE):
            continue                                    # unchanged since last send
        item = {"recording_id": r.clip_id, "model_label": label}
        if SEND_CONFIDENCE:
            item["model_confidence"] = confidence
        items.append(item)
        rows.append({"clip_id": r.clip_id, "model_label": label,
                     "confidence": confidence if SEND_CONFIDENCE else "", "fused_score": r.fused_score,
                     "fusion_version": r.fusion_version,
                     "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    for r in waiting.itertuples():
        print(f"wait  {r.clip_id}: {r.reasons}")
    if not items:
        print("Nothing new to send.")
    else:
        saved = cloud.send_labels(items)
        if SEND_CONFIDENCE:
            missing = [s["recording_id"] for s in saved if s.get("model_confidence") is None]
            if missing:
                raise SystemExit(
                    f"Backend saved {len(saved)} labels but did not store a confidence for {len(missing)} of them "
                    f"(e.g. {missing[0]}). Check the field name with the backend owner. "
                    "sent_labels.csv was NOT updated, so the next run will send these again.")
        new_file = not SENT_LABELS_CSV.exists()
        with open(SENT_LABELS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SENT_FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(rows)
        for s in saved:
            conf = f" (confidence {s['model_confidence']})" if s.get("model_confidence") is not None else ""
            print(f"sent  {s['recording_id']} -> {s['model_label']}{conf}")
        print(f"{len(saved)} labels saved in the backend.")

    if DELETE_MEDIA_AFTER_SEND:
        done = set(pd.read_csv(SENT_LABELS_CSV, dtype=str)["clip_id"]) if SENT_LABELS_CSV.exists() else set()
        n = 0
        for rid in done:
            if any(DOWNLOADS_DIR.glob(f"{rid}.*")) or (AUDIO_DIR / f"{rid}.wav").exists() or (FRAMES_DIR / rid).exists():
                cleanup(rid)
                n += 1
        if n:
            print(f"Deleted local media for {n} labelled recordings (features and CSVs kept).")
    print("\nCloud send done.")


if __name__ == "__main__":
    main()
