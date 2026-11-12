"""Shared settings for every step. Import this FIRST in each script."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # D:\Silent Cries
# "local" = your own clips in test_clips\  |  "cloud" = recordings downloaded from the backend API
PIPELINE_SOURCE = os.environ.get("PIPELINE_SOURCE", "local")
LOCAL_CLIPS_DIR = ROOT / "test_clips"
DOWNLOADS_DIR = ROOT / "work" / "downloads"
CLIPS_DIR = DOWNLOADS_DIR if PIPELINE_SOURCE == "cloud" else LOCAL_CLIPS_DIR
AUDIO_DIR = ROOT / "work" / "audio"                # extracted 16 kHz mono .wav files
OUTPUT_DIR = ROOT / "outputs"
RESULTS_CSV = OUTPUT_DIR / "predictions.csv"       # one row per clip x model (later: Postgres)

# Keep downloaded Hugging Face models on D: instead of filling C:
MODELS_CACHE = ROOT / "models_cache"
os.environ.setdefault("HF_HOME", str(MODELS_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows warning
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")        # slow connection: wait longer before retrying

SAMPLE_RATE = 16000        # both wav2vec2 models expect 16 kHz
CHUNK_SECONDS = 8.0        # Dpngtm wants < 10 s, so long clips are cut into 8 s pieces

# --- Provisional stress mappings (to be calibrated on psychologist labels later) ---
# Dpngtm outputs 7 emotions; these are summed into a "stress probability".
DPNGTM_MODEL = "Dpngtm/wav2vec2-emotion-recognition"
# Model labels are: angry, calm, disgust, fearful, happy, sad, surprised
DPNGTM_STRESS_CLASSES = {"angry", "disgust", "fearful", "sad"}
DPNGTM_STRESS_THRESHOLD = 0.5
# Bump this whenever the mapping above changes: results are stored per version,
# so old rows are kept and clips are re-scored under the new mapping.
DPNGTM_MAPPING_VERSION = "map2"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".3gp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

for d in (LOCAL_CLIPS_DIR, DOWNLOADS_DIR, AUDIO_DIR, OUTPUT_DIR, MODELS_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# --- audeering arousal / dominance / valence model ---
AUDEERING_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
# Provisional rule: stress ~ high arousal + low valence.
# stress_score = (arousal + (1 - valence)) / 2, stressed if >= threshold.
AUDEERING_STRESS_THRESHOLD = 0.5
AUDEERING_MAPPING_VERSION = "map1"
EMBEDDINGS_DIR = ROOT / "work" / "embeddings"   # saved for later fusion / fine-tuning
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# --- eGeMAPS acoustic features (openSMILE) ---
# Not a voter yet: 88 interpretable voice features per clip, used later to train a classifier
# once psychologist labels exist, and to show the psychologist pitch/jitter/shimmer plots.
EGEMAPS_CSV = OUTPUT_DIR / "egemaps_features.csv"

# --- py-feat facial model (video) ---
FRAMES_DIR = ROOT / "work" / "frames"           # frames pulled from each video
FEATURES_DIR = ROOT / "work" / "features"       # per-frame AU / emotion / V-A tables
FRAME_SAMPLE_FPS = 5                            # frames per second analysed (5 is plenty for expressions)
FRAME_WIDTH = 640                               # resize frames to save GPU memory
PYFEAT_BATCH_SIZE = 8                           # lower to 4 if you get CUDA out-of-memory
# Provisional rule, same shape as audeering: py-feat valence/arousal are in [-1, 1].
# stress_score = ((arousal + 1)/2 + (1 - valence)/2) / 2, stressed if >= threshold.
PYFEAT_STRESS_THRESHOLD = 0.5
PYFEAT_MAPPING_VERSION = "map1"
TENSION_AUS = ["AU04", "AU07", "AU23", "AU24"]  # brow lowerer, lid tightener, lip tightener/pressor
for d in (FRAMES_DIR, FEATURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Fusion (weighted vote across members) ---
FUSED_CSV = OUTPUT_DIR / "fused_labels.csv"
FUSION_WEIGHTS = {            # set a weight to 0 to drop a member; adjust after psychologist kappa
    "dpngtm_emotion": 1.0,
    "audeering_dim": 1.0,
    "pyfeat_face": 1.0,
}
FUSION_THRESHOLD = 0.5
BORDERLINE_MARGIN = 0.10      # fused score within 0.5 +/- this -> flagged for review
FUSION_VERSION = "fuse2"   # fuse2: label = weighted majority vote

# --- Cloud (backend API) ---
ENV_FILE = ROOT / ".env"                         # holds ML_API_TOKEN (never commit / share it)
API_BASE_URL_DEFAULT = "https://silent-cries-api-2026-cgfqb5b3fghwg0gj.centralindia-01.azurewebsites.net"
CLOUD_RECORDINGS_CSV = OUTPUT_DIR / "cloud_recordings.csv"   # last list fetched from the API
SENT_LABELS_CSV = OUTPUT_DIR / "sent_labels.csv"             # labels already saved in the backend
DELETE_MEDIA_AFTER_SEND = True   # delete downloaded video/audio/frames once the label is saved
AUDIO_MEMBERS = ["dpngtm_emotion", "audeering_dim"]          # run on every recording
VIDEO_MEMBERS = ["pyfeat_face"]                              # run only on video recordings
# Send a confidence score with each label. Turn on once the backend accepts the "confidence" field.
# confidence = how strongly the models support the label sent:
#   Stressed -> fused_score,  Not Stressed -> 1 - fused_score   (0..1; ~0.5 = unsure)
SEND_CONFIDENCE = True
