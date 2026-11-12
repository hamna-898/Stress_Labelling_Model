"""Step 1 check: is PyTorch installed with CUDA, and can it use the GTX 1650?"""
import platform
import sys

print("Python :", sys.version.split()[0], platform.architecture()[0])
try:
    import torch
except ImportError:
    sys.exit("FAIL: PyTorch is not installed in this environment.")

print("PyTorch:", torch.__version__, "| built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FAIL: CUDA not available. Check the NVIDIA driver (run nvidia-smi).")

name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU    : {name} ({vram:.1f} GB VRAM)")

# Small fp16 matrix multiply on the GPU to prove it actually runs.
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
b = a @ a
torch.cuda.synchronize()
print("GPU fp16 test: OK")

for pkg in ["transformers", "librosa", "soundfile", "opensmile", "pandas", "imageio_ffmpeg"]:
    try:
        mod = __import__(pkg)
        print(f"  {pkg:15s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {pkg:15s} MISSING ({e})")
print("\nStep 1 PASSED" )
