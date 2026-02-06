# YOLOv8n Inference Benchmark (CPU Baseline)

This repository contains baseline inference performance measurements for YOLOv8n
on a CPU-only environment using PyTorch.


## Benchmark Configuration

- Model: YOLOv8n
- Framework: PyTorch
- Input Resolution: 640×640
- Batch Size: 1
- Device: CPU


## Performance Results

| Metric            | Value        |
|-------------------|-------------|
| Average Latency   | 36.52 ms    |
| FPS               | 27.38       |
| RAM Usage         | 376.74 MB   |


## Conclusion

On a CPU-only environment, YOLOv8n achieves approximately 27 FPS with an average
latency of 36 ms at 640×640 resolution and batch size 1. This makes it suitable
for real-time inference on mid-range CPUs, while still requiring further
optimization for edge devices with tighter power and memory constraints.
