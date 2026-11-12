"""Step 5 check: PyTorch with CUDA + py-feat import in video_env."""
import sys

import torch

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FAIL: CUDA not available in video_env (did pip install a CPU-only torch?)")
print("GPU    :", torch.cuda.get_device_name(0))

from feat_loader import load_detector_class  # noqa: E402

Detectorv2 = load_detector_class()
import feat  # noqa: E402

print("py-feat:", feat.__version__)
print("\nStep 5 environment check PASSED")
