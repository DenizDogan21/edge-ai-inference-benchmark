# YOLOv8n ONNX Graph Analysis

## Summary
- Total nodes: 241
- Conv layers: 64
- Conv ratio: 27%
- Resize ops: 2
- Post-processing heavy ops: Sigmoid, Mul

## Potential Bottlenecks
- Resize ops may trigger memory reallocation
- Multiple Sigmoid + Mul chains → post-process overhead
- Multiple outputs enabled (should be pruned)

## Edge Notes
- Fixed input shape recommended
- Output pruning required
- Candidate for INT8 quantization
