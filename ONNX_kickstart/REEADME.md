##  ONNX Transition

- Exported YOLOv8n from PyTorch to ONNX
- Inspected ONNX graph (241 nodes, 64 Conv layers)
- Benchmarked PyTorch vs ONNX Runtime on CPU
- ONNX Runtime achieved ~33% speedup
- Observed latency jitter and multiple output overhead

Conclusion:
ONNX Runtime is preferred for edge deployment,
but further optimization is required.
