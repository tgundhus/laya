import argparse
import os
import torch

from laya.agent import Agent


def _constant(graph, name):
    """The value of an initializer or Constant output, or None."""
    from onnx import numpy_helper
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name and node.attribute:
            return numpy_helper.to_array(node.attribute[0].t)
    return None


def strip_nan_guards(path):
    """Drop the `Where(IsNaN(p), 0, p)` torch's SDPA decomposition puts after each attention softmax.

    It only fires on a fully masked row, which an additive mask of finfo.min (what transformers
    builds) never produces, yet costs two passes over every [batch, heads, seq, seq] tensor: ~13%
    of an ONNX Runtime CPU forward on a 300-token email. A guard is kept unless the softmax input
    is scores plus a `Where(mask, 0, c)` with a finite `c`.
    """
    import numpy as np
    import onnx

    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    producer = {out: node for node in graph.node for out in node.output}
    rename, dropped = {}, set()
    for node in graph.node:
        isnan = producer.get(node.input[0]) if node.op_type == "Where" else None
        if isnan is None or isnan.op_type != "IsNaN" or isnan.input[0] != node.input[2]:
            continue
        softmax = producer.get(node.input[2])
        add = producer.get(softmax.input[0]) if softmax is not None and softmax.op_type == "Softmax" else None
        masks = [producer.get(i) for i in add.input] if add is not None and add.op_type == "Add" else []
        fills = [_constant(graph, m.input[2]) for m in masks if m is not None and m.op_type == "Where"]
        if not fills or any(c is None or not np.all(np.isfinite(c)) for c in fills):
            continue
        rename[node.output[0]] = node.input[2]
        dropped.update((id(node), id(isnan)))
    if rename:
        kept = [n for n in graph.node if id(n) not in dropped]
        for n in kept:
            for i, name in enumerate(n.input):
                n.input[i] = rename.get(name, name)
        del graph.node[:]
        graph.node.extend(kept)
        onnx.save(model, path)
    return len(rename)


def quantize_int8(path, output_path):
    """Write an int8 copy of an exported model: MatMul/Gemm weights in QInt8, dynamic activations.

    About 2x faster than fp32 on an AVX-512 VNNI CPU and a third of the size, but the arithmetic
    changes, so answers near a decision boundary can change too; measure it on your own data
    (research/scripts/bench_cpu_fast_path.py) before serving it.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # The dynamo exporter leaves value_info annotations ONNX shape inference disagrees with
    # (act_head: "(1028) vs (256)"), which aborts quantize_dynamic; drop them from a scratch copy.
    model = onnx.load(path, load_external_data=False)
    del model.graph.value_info[:]
    scratch = os.path.join(os.path.dirname(os.path.abspath(path)), "_quantize_" + os.path.basename(path))
    onnx.save(model, scratch)
    try:
        quantize_dynamic(scratch, output_path, weight_type=QuantType.QInt8, per_channel=False,
                         op_types_to_quantize=["MatMul", "Gemm"])
    finally:
        os.remove(scratch)


def export_to_onnx(model_id_or_path: str, output_path: str):
    print(f"Loading PyTorch Agent from: {model_id_or_path}")
    agent = Agent(model_id_or_path, compile=False, device="cpu")
    
    print("Creating dummy input tensors...")
    # 1. Dummy tensors for tracing
    # (batch_size=2, seq_len=16): a batch of 1 lets the exporter bake batch=1 into shapes
    # (act_logits was declared [1, 2]).
    dummy_input_ids = torch.randint(0, 100, (2, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, 16), dtype=torch.long)
    
    # (batch_size=2, num_markers=2)
    dummy_marker_pos = torch.tensor([[1, 5], [1, 5]], dtype=torch.long)
    dummy_marker_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
    
    # (batch_size=2)
    dummy_qtype = torch.tensor([0, 0], dtype=torch.long)
    
    inputs = (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_marker_pos,
        dummy_marker_mask,
        dummy_qtype,
    )

    # 2. Define dynamic axes so the model can accept variable batch sizes and sequence lengths
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }

    input_names = [
        "input_ids",
        "attention_mask",
        "marker_pos",
        "marker_mask",
        "qtype",
    ]
    
    output_names = ["logits", "act_logits"]

    print(f"Exporting to {output_path} (this may take a minute)...")
    
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # We must detach the encoder because ONNX export runs the model in trace mode.
    # The `detach_encoder` flag in forward() just detaches the hidden state gradient, 
    # but we don't even need to pass it since kwargs are ignored by tracing.
    
    torch.onnx.export(
        agent.model,
        inputs,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Removed {strip_nan_guards(output_path)} redundant attention NaN guards")
    print(f"Successfully exported ONNX model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Laya model to ONNX format")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="HuggingFace Hub ID or local path")
    parser.add_argument("--output", type=str, default="laya.onnx", help="Output path for the ONNX file")
    parser.add_argument("--int8", action="store_true",
                        help="also write an int8 copy next to it (<output>.int8.onnx); answers can change")
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)
    if args.int8:
        int8_path = os.path.splitext(args.output)[0] + ".int8.onnx"
        quantize_int8(args.output, int8_path)
        print(f"Wrote the int8 copy to: {int8_path}")
