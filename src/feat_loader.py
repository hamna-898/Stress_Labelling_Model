"""Import py-feat safely on Windows.

py-feat imports `torchcodec` (a video decoder that needs separate FFmpeg DLLs on Windows).
We never give py-feat a video -- we extract frames ourselves with ffmpeg -- so if torchcodec
can't load, we put a placeholder in its place and py-feat works normally on images.
"""
import sys
import types


def load_detector_class():
    try:
        import torchcodec.decoders  # noqa: F401
        print("torchcodec: loaded")
    except Exception as e:  # noqa: BLE001
        for name in [m for m in sys.modules if m == "torchcodec" or m.startswith("torchcodec.")]:
            del sys.modules[name]
        tc = types.ModuleType("torchcodec")
        dec = types.ModuleType("torchcodec.decoders")

        class VideoDecoder:  # placeholder: only used if someone passes a video file to py-feat
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Video decoding is disabled here; pass extracted frames instead.")

        dec.VideoDecoder = VideoDecoder
        tc.decoders = dec
        sys.modules["torchcodec"] = tc
        sys.modules["torchcodec.decoders"] = dec
        print(f"torchcodec: not usable ({type(e).__name__}) - fine, we analyse extracted frames")
    from feat import Detectorv2
    return Detectorv2
