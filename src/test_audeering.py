"""Step 3: run the audeering wav2vec2 arousal/dominance/valence model on every extracted .wav.

Outputs per clip: arousal, dominance, valence (roughly 0..1), a provisional stress score,
and a 1024-number embedding saved to work\\embeddings\\ for later fusion.
First run downloads the model (~800 MB) into models_cache\\.
Licence: CC-BY-NC-SA 4.0 -- research use only.
Run:  python src\\test_audeering.py
"""
import time

import config  # noqa: F401  (must come first: sets HF_HOME)
from config import (AUDEERING_MAPPING_VERSION, AUDEERING_MODEL, AUDEERING_STRESS_THRESHOLD,
                    AUDIO_DIR, CHUNK_SECONDS, EMBEDDINGS_DIR, SAMPLE_RATE)

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from common import already_done, load_audio, save_prediction, split_chunks

MEMBER = "audeering_dim"


# ---- Model classes, as given on the audeering model card ----
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.post_init()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]       # (batch, time, 1024)
        pooled = hidden.mean(dim=1)                   # average over time -> embedding
        return pooled, self.classifier(pooled)        # logits order: arousal, dominance, valence


def main():
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print("No .wav files. Run: python src\\extract_audio.py")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {AUDEERING_MODEL} on {device} (first time downloads ~800 MB)...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDEERING_MODEL)
    model = EmotionModel.from_pretrained(AUDEERING_MODEL).to(device).eval()
    version = f"{AUDEERING_MODEL}@{getattr(model.config, '_commit_hash', None) or 'unknown'}|{AUDEERING_MAPPING_VERSION}"

    for wav in wavs:
        clip_id = wav.stem
        if already_done(clip_id, MEMBER, version):
            print(f"skip  {clip_id} (already in predictions.csv)")
            continue
        t0 = time.time()
        wave = load_audio(wav)
        chunks = split_chunks(wave, CHUNK_SECONDS)
        scores, embs, weights = [], [], []
        for chunk in chunks:
            x = extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                emb, logits = model(x)
            scores.append(logits.float()[0].cpu().numpy())
            embs.append(emb.float()[0].cpu().numpy())
            weights.append(len(chunk))
        arousal, dominance, valence = np.average(np.stack(scores), axis=0, weights=weights)
        embedding = np.average(np.stack(embs), axis=0, weights=weights)
        np.save(EMBEDDINGS_DIR / f"{clip_id}__{MEMBER}.npy", embedding.astype(np.float32))

        stress_score = float(np.clip((arousal + (1 - valence)) / 2, 0, 1))
        vote = stress_score >= AUDEERING_STRESS_THRESHOLD
        save_prediction(clip_id, MEMBER, version,
                        {"arousal": round(float(arousal), 4), "dominance": round(float(dominance), 4),
                         "valence": round(float(valence), 4), "stress_score": round(stress_score, 4),
                         "n_chunks": len(chunks),
                         "embedding_file": f"work/embeddings/{clip_id}__{MEMBER}.npy"},
                        vote, stress_score)
        print(f"ok    {clip_id}: arousal={arousal:.2f} dominance={dominance:.2f} valence={valence:.2f}  "
              f"stress_score={stress_score:.2f} -> {'STRESSED' if vote else 'not stressed'}  "
              f"[{len(wave)/SAMPLE_RATE:.1f}s audio, {time.time()-t0:.1f}s]")

    if device == "cuda":
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB of 4096 MB")
    print("Step 3 done. Results in outputs\\predictions.csv, embeddings in work\\embeddings\\")


if __name__ == "__main__":
    main()
