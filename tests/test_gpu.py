"""
PyTorch NVIDIA GPU Diagnostic Script
Run: python pytorch_gpu_test.py
"""
import torch
import sys

def separator(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# ── 1. Basic environment info
separator("Environment")
print(f"Python   : {sys.version.split()[0]}")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n[FAIL] CUDA not available. Check:")
    print("  • PyTorch is installed with CUDA support")
    print("  • NVIDIA drivers are up to date")
    print("  • Run: nvidia-smi")
    sys.exit(1)

# ── 2. GPU inventory
separator("GPU Inventory")
n = torch.cuda.device_count()
print(f"GPUs found: {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f"  GPU {i}: {props.name}")
    print(f"         VRAM    : {mem_gb:.1f} GB")
    print(f"         Compute : {props.major}.{props.minor}")
    print(f"         SMs     : {props.multi_processor_count}")

print(f"\nCUDA version (runtime): {torch.version.cuda}")
print(f"cuDNN version         : {torch.backends.cudnn.version()}")
print(f"cuDNN enabled         : {torch.backends.cudnn.enabled}")

# ── 3. Tensor operations on each GPU
separator("Tensor Operations")
for i in range(n):
    device = torch.device(f"cuda:{i}")
    try:
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(device)
        print(f"  [OK] GPU {i}: 1000x1000 matmul passed (mean={c.mean().item():.4f})")
    except Exception as e:
        print(f"  [FAIL] GPU {i}: {e}")

# ── 4. Memory allocation
separator("Memory Allocation")
for i in range(n):
    device = torch.device(f"cuda:{i}")
    try:
        t = torch.zeros(512, 512, 512, device=device)  # ~512 MB
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        print(f"  [OK] GPU {i}: allocated={alloc:.0f} MB  reserved={reserved:.0f} MB")
        del t
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [FAIL] GPU {i}: {e}")

# ── 5. Autograd (backprop)
separator("Autograd / Backprop")
for i in range(n):
    device = torch.device(f"cuda:{i}")
    try:
        x = torch.randn(100, 100, device=device, requires_grad=True)
        loss = (x ** 2).sum()
        loss.backward()
        print(f"  [OK] GPU {i}: backward pass OK (grad norm={x.grad.norm().item():.2f})")
    except Exception as e:
        print(f"  [FAIL] GPU {i}: {e}")

# ── 6. GPU-to-GPU bandwidth (if multiple GPUs)
if n > 1:
    separator("GPU-to-GPU Transfer")
    import time
    src = torch.randn(10_000, 10_000, device="cuda:0")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dst = src.to("cuda:1")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mb = src.nbytes() / 1024**2
    print(f"  cuda:0 → cuda:1 : {mb:.0f} MB in {elapsed*1000:.1f} ms")
    print(f"  Bandwidth       : {mb/elapsed/1024:.1f} GB/s")

# ── 7. Mixed precision (AMP)
separator("Mixed Precision (AMP)")
try:
    device = torch.device("cuda:0")
    with torch.amp.autocast("cuda"):
        a = torch.randn(512, 512, device=device)
        b = torch.matmul(a, a.T)
    print(f"  [OK] AMP autocast OK (dtype={b.dtype})")
except Exception as e:
    print(f"  [SKIP] AMP not supported: {e}")

# ── 8. Benchmark (GFLOPS)
separator("Benchmark (GFLOPS)")
import time
for i in range(n):
    device = torch.device(f"cuda:{i}")
    N = 4096
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    for _ in range(3): torch.matmul(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(10): torch.matmul(a, b)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    flops = 2 * N**3 * 10 / elapsed / 1e12
    print(f"  GPU {i}: ~{flops:.1f} TFLOPS (FP32, {N}x{N} matmul)")

separator("All Tests Complete")
print("  [OK] marks = passed   [FAIL] = needs attention\n")