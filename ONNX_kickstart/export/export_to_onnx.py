import torch
from ultralytics import YOLO

# 1. Load PyTorch model
model = YOLO("models/yolov8n.pt")
model.model.eval()

# 2. Dummy input (batch=1, 640x640)
dummy_input = torch.randn(1, 3, 640, 640)

# 3. Export to ONNX
torch.onnx.export(
    model.model,
    dummy_input,
    "models/yolov8n.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

print(" YOLOv8n exported to ONNX successfully")
