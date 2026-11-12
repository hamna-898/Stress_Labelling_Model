"""Step 6: combine every member's vote into one label per clip (late fusion / weighted vote).

Reads outputs\\predictions.csv (latest row per clip x member), and writes outputs\\fused_labels.csv with:
  fused_score   weighted mean of the members' stress scores (0..1), kept for information
  fused_label   weighted majority vote: 1 = stressed, 0 = not stressed (ties -> fused_score)
  vote_share    weighted share of members voting "stressed"
  agreement     share of members whose vote matches the fused label
  review_priority  'high' if members disagree, the score is borderline, or a member is missing
  vote_* / score_*  each member's own output, kept next to the combined label (requirement 4.3)
Pure pandas: runs in either environment.
Run:  python src\\fuse.py
"""
from datetime import datetime, timezone

import config  # noqa: F401
from config import (AUDIO_MEMBERS, BORDERLINE_MARGIN, CLOUD_RECORDINGS_CSV, DOWNLOADS_DIR, FUSED_CSV,
                    FUSION_THRESHOLD, FUSION_VERSION, FUSION_WEIGHTS, LOCAL_CLIPS_DIR, RESULTS_CSV,
                    VIDEO_EXTS, VIDEO_MEMBERS)

import pandas as pd


def clip_modalities():
    """clip_id -> 'audio' / 'video': from the backend list for cloud recordings, else from the file type."""
    out = {}
    for folder in (LOCAL_CLIPS_DIR, DOWNLOADS_DIR):
        for p in folder.glob("*.*"):
            out[p.stem] = "video" if p.suffix.lower() in VIDEO_EXTS else "audio"
    if CLOUD_RECORDINGS_CSV.exists():
        for r in pd.read_csv(CLOUD_RECORDINGS_CSV, dtype=str).itertuples():
            out[r.recording_id] = str(r.modality).lower()   # backend may send "Video" or "video"
    return out


def main():
    if not RESULTS_CSV.exists():
        print("No predictions.csv yet - run the model scripts first.")
        return
    active = [m for m, w in FUSION_WEIGHTS.items() if w > 0]
    allpred = pd.read_csv(RESULTS_CSV, dtype={"clip_id": str})
    allpred = allpred[allpred["member"].isin(active)]
    allpred = allpred.sort_values("created_at").groupby(["clip_id", "member"]).tail(1)  # latest per member
    attempted = allpred.groupby("clip_id")["member"].apply(set).to_dict()  # ran, with or without a vote
    pred = allpred.dropna(subset=["confidence"])                          # rows that actually voted
    modality = clip_modalities()

    rows = []
    for clip_id, g in pred.groupby("clip_id"):
        g = g.set_index("member")
        is_video = modality.get(clip_id) == "video"
        expected = [m for m in AUDIO_MEMBERS + (VIDEO_MEMBERS if is_video else []) if m in active]
        weights = pd.Series({m: FUSION_WEIGHTS[m] for m in g.index})
        scores = g["confidence"].astype(float)
        fused = float((scores * weights).sum() / weights.sum())      # average score (information)
        votes = g["stress_vote"].astype(int)
        vote_share = float((votes * weights).sum() / weights.sum())  # weighted share voting "stressed"
        # Label = weighted MAJORITY of votes, so one over-confident member can't outvote the rest.
        # Exact tie (e.g. 2 vs 2) -> fall back to the average score.
        if vote_share > 0.5:
            label = 1
        elif vote_share < 0.5:
            label = 0
        else:
            label = int(fused >= FUSION_THRESHOLD)
        agreement = float((votes == label).mean())
        not_run = [m for m in expected if m not in attempted.get(clip_id, set())]
        no_vote = [m for m in expected if m in attempted.get(clip_id, set()) and m not in g.index]

        reasons = []
        if agreement < 1:
            reasons.append("members disagree")
        if abs(fused - FUSION_THRESHOLD) < BORDERLINE_MARGIN:
            reasons.append("borderline score")
        if int(fused >= FUSION_THRESHOLD) != label:
            reasons.append("average score contradicts majority")
        if not_run:
            reasons.append("not run yet: " + ",".join(not_run))
        if no_vote:
            reasons.append("no vote from: " + ",".join(no_vote))

        row = {"clip_id": clip_id, "modality": modality.get(clip_id, "unknown"), "fused_score": round(fused, 4), "fused_label": label, "vote_share": round(vote_share, 3),
               "n_members": len(g), "agreement": round(agreement, 3),
               "review_priority": "high" if reasons else "normal", "reasons": "; ".join(reasons)}
        for m in active:
            row[f"vote_{m}"] = int(votes[m]) if m in votes else None
            row[f"score_{m}"] = round(float(scores[m]), 4) if m in scores else None
        row["fusion_version"] = FUSION_VERSION
        row["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(FUSED_CSV, index=False)

    print(f"{'clip':15s} {'fused':>6s}  label         agree  review")
    for r in rows:
        name = "STRESSED" if r["fused_label"] else "not stressed"
        print(f"{r['clip_id']:15s} {r['fused_score']:6.2f}  {name:12s}  {r['agreement']:.2f}   "
              f"{r['review_priority']}{' (' + r['reasons'] + ')' if r['reasons'] else ''}")
    print(f"\nStep 6 done. {len(rows)} clips -> {FUSED_CSV}")


if __name__ == "__main__":
    main()
