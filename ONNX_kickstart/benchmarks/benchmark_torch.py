import time
import torch
from ultralytics import YOLO
import numpy as np

# --------------------
# Config
# --------------------
NUM_RUNS = 50
WARMUP = 10
IMG_SIZE = 640

# --------------------
# Model
# --------------------
model = YOLO("models/yolov8n.pt")
model.fuse()
model.model.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# --------------------
# Warmup
# --------------------
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model.model(dummy_input)

# --------------------
# Benchmark
# --------------------
times = []

with torch.no_grad():
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        _ = model.model(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)

times = np.array(times)

print("🟢 PyTorch Benchmark")
print(f"Avg latency : {times.mean():.2f} ms")
print(f"Min latency : {times.min():.2f} ms")
print(f"Max latency : {times.max():.2f} ms")
print(f"FPS         : {1000 / times.mean():.2f}")
