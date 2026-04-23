"""GPU keeper — keeps all GPUs busy continuously to block BSC idle-suspend.

Runs 2048x2048 fp16 matmul loops on every GPU with 2s cadence. Memory footprint
is ~8 MB per GPU (negligible vs 16GB SDC model). Designed to coexist with real
training: when SDC allocates the bulk of GPU memory, keeper tensors are already
resident and keep running; compute overhead is <1% of an H200.

Exits when /scratch/gpu_keeper.stop exists. Orchestrator never sets stop — keeper
runs for the lifetime of the job.
"""
import os, time

STOP = "/scratch/gpu_keeper.stop"
try:
    os.remove(STOP)
except FileNotFoundError:
    pass

import torch

n_gpus = torch.cuda.device_count()
print(f"[gpu_keeper] n_gpus={n_gpus}", flush=True)

tensors = [torch.randn(2048, 2048, device=f"cuda:{g}", dtype=torch.float16) for g in range(n_gpus)]

i = 0
while not os.path.exists(STOP):
    for g, t in enumerate(tensors):
        tensors[g] = (t @ t) * 0.5 + torch.randn_like(t) * 0.01
    if i % 30 == 0:
        mem = [torch.cuda.memory_allocated(g) / 1e9 for g in range(n_gpus)]
        print(f"[gpu_keeper] iter={i} allocated_gb={mem}", flush=True)
    i += 1
    time.sleep(2)

print("[gpu_keeper] stop-file seen — exiting", flush=True)
