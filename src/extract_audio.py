"""Step 2a: turn every clip in test_clips\\ into a 16 kHz mono .wav in work\\audio\\.

Uses the ffmpeg that comes with the imageio-ffmpeg package, so nothing extra to install.
Run:  python src\\extract_audio.py
"""
import subprocess

import imageio_ffmpeg

from config import AUDIO_DIR, AUDIO_EXTS, CLIPS_DIR, SAMPLE_RATE, VIDEO_EXTS

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    clips = sorted(p for p in CLIPS_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS | AUDIO_EXTS)
    if not clips:
        print(f"No clips found. Put a few video files in: {CLIPS_DIR}")
        return
    ok = 0
    for clip in clips:
        out = AUDIO_DIR / f"{clip.stem}.wav"
        if out.exists():
            print(f"skip  {clip.name} (already extracted)")
            ok += 1
            continue
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(clip),
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAIL  {clip.name}: {result.stderr.strip()[:300]}")
            continue
        print(f"ok    {clip.name} -> {out.relative_to(CLIPS_DIR.parent)}")
        ok += 1
    print(f"\nStep 2a: {ok}/{len(clips)} clips have audio in {AUDIO_DIR}")


if __name__ == "__main__":
    main()
