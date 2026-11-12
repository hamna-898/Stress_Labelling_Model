"""Step 2b: run the Dpngtm wav2vec2 emotion model on every extracted .wav.

First run downloads the model (~380 MB) into models_cache\\.
Run:  python src\\test_dpngtm.py
"""
import time

import config  # noqa: F401  (must come first: sets HF_HOME)
from config import AUDIO_DIR, CHUNK_SECONDS, DPNGTM_MODEL, DPNGTM_STRESS_CLASSES, DPNGTM_STRESS_THRESHOLD, DPNGTM_MAPPING_VERSION, SAMPLE_RATE

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from common import already_done, load_audio, save_prediction, split_chunks

MEMBER = "dpngtm_emotion"


def main():
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print("No .wav files. Run: python src\\extract_audio.py")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {DPNGTM_MODEL} on {device} (first time downloads it)...")
    extractor = AutoFeatureExtractor.from_pretrained(DPNGTM_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(DPNGTM_MODEL).to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    version = f"{DPNGTM_MODEL}@{getattr(model.config, '_commit_hash', None) or 'unknown'}|{DPNGTM_MAPPING_VERSION}"
    print("Model classes:", labels)
    unknown = DPNGTM_STRESS_CLASSES - set(labels)
    if unknown:
        print("WARNING: stress classes not in model labels:", unknown)

    for wav in wavs:
        clip_id = wav.stem
        if already_done(clip_id, MEMBER, version):
            print(f"skip  {clip_id} (already in predictions.csv)")
            continue
        t0 = time.time()
        wave = load_audio(wav)
        chunks = split_chunks(wave, CHUNK_SECONDS)
        probs, weights = [], []
        for chunk in chunks:
            inputs = extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**inputs).logits
            probs.append(torch.softmax(logits.float(), dim=-1)[0].cpu().numpy())
            weights.append(len(chunk))
        p = np.average(np.stack(probs), axis=0, weights=weights)   # length-weighted average over chunks

        emotions = {lab: round(float(v), 4) for lab, v in zip(labels, p)}
        stress_prob = sum(v for lab, v in emotions.items() if lab in DPNGTM_STRESS_CLASSES)
        vote = stress_prob >= DPNGTM_STRESS_THRESHOLD
        save_prediction(clip_id, MEMBER, version,
                        {"emotions": emotions, "stress_prob": round(stress_prob, 4), "n_chunks": len(chunks)},
                        vote, stress_prob)
        top = max(emotions, key=emotions.get)
        print(f"ok    {clip_id}: top={top} ({emotions[top]:.2f})  stress_prob={stress_prob:.2f}  "
              f"-> {'STRESSED' if vote else 'not stressed'}  [{len(wave)/SAMPLE_RATE:.1f}s audio, {time.time()-t0:.1f}s]")

    if device == "cuda":
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB of 4096 MB")
    print("Step 2b done. Results in outputs\\predictions.csv")


if __name__ == "__main__":
    main()
