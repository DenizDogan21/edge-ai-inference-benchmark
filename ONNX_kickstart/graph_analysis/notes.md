## Day 3 – PyTorch vs ONNX

- PyTorch avg latency: 30.3 ms (~33 FPS)
- ONNX Runtime avg latency: 20.3 ms (~49 FPS)

ONNX Runtime provides ~33% speedup on CPU.

However, ONNX inference shows higher jitter (max latency spikes),
which may affect real-time edge deployments.

Conclusion:
ONNX Runtime is preferred, but further optimization
(quantization / fixed shapes / OpenVINO) is required for stable FPS.
