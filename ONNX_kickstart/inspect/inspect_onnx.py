import onnx
from collections import Counter

model = onnx.load("models/yolov8n.onnx")
graph = model.graph

print("🔹 Model name:", graph.name)
print("🔹 Total nodes:", len(graph.node))

# Node type distribution
op_types = [node.op_type for node in graph.node]
counter = Counter(op_types)

print("\n🔹 Node type counts:")
for op, count in counter.most_common():
    print(f"{op:15s}: {count}")

conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
print(f"\n Conv layers: {len(conv_nodes)}")
print(f" Conv ratio: {len(conv_nodes) / len(graph.node):.2f}")

def get_shape(value_info):
    return [
        dim.dim_value if dim.dim_value > 0 else "?"
        for dim in value_info.type.tensor_type.shape.dim
    ]

print("\n🔹 Inputs:")
for inp in graph.input:
    print(inp.name, get_shape(inp))

print("\n🔹 Outputs:")
for out in graph.output:
    print(out.name, get_shape(out))
