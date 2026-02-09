import time
import onnxruntime as ort
import numpy as np

# --------------------
# Config
# --------------------
NUM_RUNS = 50
WARMUP = 10
IMG_SIZE = 640

# --------------------
# Session
# --------------------
session = ort.InferenceSession(
    "models/yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

# --------------------
# Warmup
# --------------------
for _ in range(WARMUP):
    _ = session.run(None, {input_name: dummy_input})

# --------------------
# Benchmark
# --------------------
times = []

for _ in range(NUM_RUNS):
    start = time.perf_counter()
    _ = session.run(None, {input_name: dummy_input})
    end = time.perf_counter()
    times.append((end - start) * 1000)

times = np.array(times)

print("🟦 ONNX Runtime Benchmark")
print(f"Avg latency : {times.mean():.2f} ms")
print(f"Min latency : {times.min():.2f} ms")
print(f"Max latency : {times.max():.2f} ms")
print(f"FPS         : {1000 / times.mean():.2f}")
