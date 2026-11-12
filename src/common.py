"""Helpers shared by all model scripts.

save_prediction() writes to a local CSV for now. When we connect the cloud,
only this function changes to insert into Postgres -- the model code stays the same.
"""
import csv
import json
from datetime import datetime, timezone

import numpy as np

from config import RESULTS_CSV, SAMPLE_RATE

FIELDS = ["clip_id", "member", "member_version", "output_json",
          "stress_vote", "confidence", "created_at"]


def already_done(clip_id: str, member: str, member_version: str) -> bool:
    """True if this clip was already processed by this exact model version (resume support)."""
    if not RESULTS_CSV.exists():
        return False
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["clip_id"], row["member"], row["member_version"]) == (clip_id, member, member_version):
                return True
    return False


def save_prediction(clip_id, member, member_version, output: dict, stress_vote, confidence):
    """Append one result row immediately, so a crash never loses finished clips."""
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "clip_id": clip_id,
            "member": member,
            "member_version": member_version,
            "output_json": json.dumps(output),
            "stress_vote": "" if stress_vote is None else int(bool(stress_vote)),
            "confidence": "" if confidence is None else round(float(confidence), 4),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


def load_audio(path) -> np.ndarray:
    """Load a 16 kHz mono wav produced by extract_audio.py as float32."""
    import soundfile as sf  # imported here so video_env (no soundfile) can still use common.py

    wave, sr = sf.read(str(path), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz, expected {SAMPLE_RATE}. Re-run extract_audio.py")
    return wave


def split_chunks(wave: np.ndarray, seconds: float, min_seconds: float = 1.0):
    """Cut audio into fixed-length pieces; drop a tiny leftover tail (unless it's the only piece)."""
    size = int(seconds * SAMPLE_RATE)
    pieces = [wave[i:i + size] for i in range(0, len(wave), size)]
    keep = [p for p in pieces if len(p) >= min_seconds * SAMPLE_RATE]
    return keep or pieces[:1]
