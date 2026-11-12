"""Step 5: run py-feat (Detectorv2) on every video in test_clips\\.

For each clip: extract 5 frames/second with ffmpeg -> detect the face -> per frame get
20 action units, 7 emotions, and facial valence/arousal. Then average over frames.
First run downloads py-feat's face models from Hugging Face.
Run (in video_env):  python src\\test_pyfeat.py
"""
import subprocess
import time

import config  # noqa: F401  (must come first: sets HF_HOME)
from config import (CLIPS_DIR, FEATURES_DIR, FRAME_SAMPLE_FPS, FRAME_WIDTH, FRAMES_DIR,
                    PYFEAT_BATCH_SIZE, PYFEAT_MAPPING_VERSION, PYFEAT_STRESS_THRESHOLD,
                    TENSION_AUS, VIDEO_EXTS)

import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch

from common import already_done, save_prediction
from feat_loader import load_detector_class

MEMBER = "pyfeat_face"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def extract_frames(clip):
    """Save FRAME_SAMPLE_FPS frames per second as JPGs; reuse them if already extracted."""
    out_dir = FRAMES_DIR / clip.stem
    frames = sorted(out_dir.glob("f_*.jpg"))
    if frames:
        return frames
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(clip),
           "-vf", f"fps={FRAME_SAMPLE_FPS},scale={FRAME_WIDTH}:-2", "-q:v", "2",
           str(out_dir / "f_%05d.jpg")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    return sorted(out_dir.glob("f_*.jpg"))


def main():
    clips = sorted(p for p in CLIPS_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not clips:
        print(f"No videos in {CLIPS_DIR}")
        return

    Detectorv2 = load_detector_class()
    import feat
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading py-feat {feat.__version__} Detectorv2 on {device} (first time downloads models)...")
    # amp=False: py-feat defaults to bfloat16 on CUDA, which the GTX 1650 (Turing) cannot run
    # ("unable to find an engine"). Full 32-bit precision works and fits in 4 GB.
    detector = Detectorv2(device=device, identity_model=None, amp=False)
    version = f"py-feat=={feat.__version__}:Detectorv2|{PYFEAT_MAPPING_VERSION}"

    for clip in clips:
        clip_id = clip.stem
        if already_done(clip_id, MEMBER, version):
            print(f"skip  {clip_id} (already in predictions.csv)")
            continue
        t0 = time.time()
        frames = extract_frames(clip)
        fex = detector.detect([str(f) for f in frames], data_type="image",
                              batch_size=PYFEAT_BATCH_SIZE, progress_bar=True)
        df = pd.DataFrame(fex)

        # Column names (match case-insensitively in case the checkpoint changes them)
        au_cols = [c for c in df.columns if str(c).startswith("AU") and str(c)[2:].isdigit()]
        emo_names = {"neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"}
        emo_cols = [c for c in df.columns if str(c).lower() in emo_names]

        # Keep frames where a face was found; if several faces, keep the most confident one.
        df["frame_idx"] = df["frame"].astype(int)
        faces = df.dropna(subset=["FaceScore", "valence", "arousal"])
        faces = faces.sort_values("FaceScore", ascending=False).drop_duplicates("frame_idx").sort_values("frame_idx")
        detect_rate = len(faces) / max(len(frames), 1)

        per_frame = faces[["frame_idx", "FaceScore", "valence", "arousal"] + emo_cols + au_cols].copy()
        per_frame.insert(1, "time_s", per_frame["frame_idx"] / FRAME_SAMPLE_FPS)
        frames_file = FEATURES_DIR / f"{clip_id}__{MEMBER}_frames.csv"
        per_frame.to_csv(frames_file, index=False)

        if faces.empty:
            save_prediction(clip_id, MEMBER, version,
                            {"error": "no face detected", "n_frames": len(frames)}, None, None)
            print(f"WARN  {clip_id}: no face detected in {len(frames)} frames - no vote")
            continue

        valence = float(faces["valence"].mean())
        arousal = float(faces["arousal"].mean())
        aus = {c: round(float(faces[c].mean()), 4) for c in au_cols}
        emotions = {str(c).lower(): round(float(faces[c].mean()), 4) for c in emo_cols}
        tension = float(np.mean([aus[a] for a in TENSION_AUS if a in aus])) if aus else None
        stress_score = float(np.clip(((arousal + 1) / 2 + (1 - valence) / 2) / 2, 0, 1))
        vote = stress_score >= PYFEAT_STRESS_THRESHOLD

        save_prediction(clip_id, MEMBER, version, {
            "valence": round(valence, 4), "arousal": round(arousal, 4),
            "stress_score": round(stress_score, 4),
            "tension_index": None if tension is None else round(tension, 4),
            "emotions": emotions, "aus": aus,
            "n_frames": len(frames), "face_detect_rate": round(detect_rate, 3),
            "frames_file": f"work/features/{frames_file.name}",
        }, vote, stress_score)
        top = max(emotions, key=emotions.get) if emotions else "?"
        print(f"ok    {clip_id}: valence={valence:+.2f} arousal={arousal:+.2f} stress_score={stress_score:.2f} "
              f"-> {'STRESSED' if vote else 'not stressed'} | top emotion={top} | tension AUs={tension:.2f} | "
              f"face in {detect_rate:.0%} of {len(frames)} frames  [{time.time()-t0:.1f}s]")

    if device == "cuda":
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB of 4096 MB")
    print("Step 5 done. Results in outputs\\predictions.csv, per-frame tables in work\\features\\")


if __name__ == "__main__":
    main()
