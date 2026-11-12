# Silent Cries – labelling pipeline (local)

Two Python environments:
- `audio_env` – PyTorch 2.7.1 (CUDA 12.6), wav2vec2 models, openSMILE
- `video_env` – PyTorch 2.11+ (CUDA 12.6), py-feat 2.1.5

## One-time setup
```
.\setup_audio_env.bat
.\setup_video_env.bat
```

## Run order (from D:\Silent Cries)
| Step | Env | Command | Output |
|---|---|---|---|
| 2a | audio | `python src\extract_audio.py` | `work\audio\*.wav` |
| 2b | audio | `python src\test_dpngtm.py` | rows in `outputs\predictions.csv` |
| 3 | audio | `python src\test_audeering.py` | rows + `work\embeddings\*.npy` |
| 4 | audio | `python src\test_egemaps.py` | `outputs\egemaps_features.csv` |
| 5 | video | `python src\test_pyfeat.py` | rows + `work\features\*_frames.csv` |
| 6 | either | `python src\fuse.py` | `outputs\fused_labels.csv` |

Switch environment: `.\audio_env\Scripts\Activate.ps1` or `.\video_env\Scripts\Activate.ps1`.
Every script skips clips already processed, so re-running is safe.

## Voting members
| member | modality | output | provisional stress rule |
|---|---|---|---|
| `dpngtm_emotion` | audio | 7 emotions | angry+disgust+fearful+sad ≥ 0.5 |
| `audeering_dim` | audio | arousal/dominance/valence (0..1) | (arousal + 1 − valence)/2 ≥ 0.5 |
| `pyfeat_face` | video | 20 AUs, 7 emotions, valence/arousal (−1..1) | ((a+1)/2 + (1−v)/2)/2 ≥ 0.5 |

eGeMAPS features are stored but do not vote until a classifier is trained on psychologist labels.
All rules, thresholds and fusion weights live in `src\config.py`. Bump the `*_MAPPING_VERSION`
when changing a rule so old results are kept and clips are re-scored.

## Cloud (backend API)
1. Put the ML token in `.env`:  `ML_API_TOKEN=<token from the backend owner>`  (never share or commit `.env`)
2. Test the connection (read-only):  `audio_env\Scripts\python.exe src\cloud_check.py`
3. Run everything:  `.\run_cloud.bat`   (first test: `.\run_cloud.bat --limit 2`)

`run_cloud.bat`: download unlabelled recordings → audio models → py-feat (video only) → fuse →
send `Stressed` / `Not Stressed` to `POST /api/ml/predictions` → delete the downloaded media.
The backend stores only the final label; every model's output stays in `outputs\predictions.csv`,
`outputs\fused_labels.csv` and `outputs\sent_labels.csv`.

## Later
- Fine-tuning: train a fusion model on `work\embeddings` + py-feat features using the psychologist-validated labels.
