"""Step 4: extract the 88 eGeMAPS voice features for every extracted .wav (CPU only, no download).

eGeMAPS is the standard acoustic feature set from the voice-stress literature:
pitch (F0) level and variability, jitter, shimmer, HNR, loudness, speech rate, etc.
Saved as one row per clip in outputs\\egemaps_features.csv.
Run:  python src\\test_egemaps.py
"""
import config  # noqa: F401
from config import AUDIO_DIR, EGEMAPS_CSV

import opensmile
import pandas as pd

# A few features that are easy to interpret, printed for a quick sanity check.
KEY_FEATURES = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "pitch (semitones)",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "pitch variability",
    "jitterLocal_sma3nz_amean": "jitter",
    "shimmerLocaldB_sma3nz_amean": "shimmer (dB)",
    "HNRdBACF_sma3nz_amean": "HNR (dB)",
    "loudness_sma3_amean": "loudness",
    "VoicedSegmentsPerSec": "voiced segments/s (speech rate)",
}


def main():
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print("No .wav files. Run: python src\\extract_audio.py")
        return

    existing = pd.read_csv(EGEMAPS_CSV) if EGEMAPS_CSV.exists() else pd.DataFrame(columns=["clip_id"])
    done = set(existing["clip_id"].astype(str))

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    rows = []
    for wav in wavs:
        clip_id = wav.stem
        if clip_id in done:
            print(f"skip  {clip_id} (already in egemaps_features.csv)")
            continue
        feats = smile.process_file(str(wav)).reset_index(drop=True).iloc[0]
        row = {"clip_id": clip_id, **feats.to_dict()}
        rows.append(row)
        summary = "  ".join(f"{label}={row[name]:.3g}" for name, label in KEY_FEATURES.items())
        print(f"ok    {clip_id}: {len(feats)} features | {summary}")

    if rows:
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(EGEMAPS_CSV, index=False)
    print(f"\nStep 4 done. {len(done) + len(rows)} clips in {EGEMAPS_CSV}")


if __name__ == "__main__":
    main()
