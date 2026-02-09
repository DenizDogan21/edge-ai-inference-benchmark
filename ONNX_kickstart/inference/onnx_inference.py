import onnxruntime as ort
import numpy as np
import time

session = ort.InferenceSession(
    "models/yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Warmup
for _ in range(5):
    session.run(None, {input_name: dummy_input})

# Measure
runs = 20
start = time.time()
for _ in range(runs):
    session.run(None, {input_name: dummy_input})
end = time.time()

print(f"⏱️ Avg ONNX inference time: {(end - start) / runs * 1000:.2f} ms")
