#!/usr/bin/env python3
"""
TensorRT Engine Builder Script
Builds TensorRT engines from ONNX models using Python API
"""
import tensorrt as trt
import os
import sys
import json
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_config(config_path="src/config.json"):
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def build_engine(onnx_path, engine_path, batch_size=256, seq_len=32, use_fp16=True):
    """
    Build TensorRT engine from ONNX model
    """
    precision = "FP16" if use_fp16 else "FP32"
    print(f"Building TensorRT engine from {onnx_path}...")
    print(f"Target engine: {engine_path}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Precision: {precision}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Set precision
    if use_fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)
    elif use_fp16:
        print("Warning: FP16 requested but not supported, using FP32")

    # Set memory pool limit (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    # Set dynamic input shapes for BERT-like models
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    for name in input_names:
        # Get the input tensor
        input_tensor = network.get_input(next(i for i in range(network.num_inputs)
                                             if network.get_input(i).name == name))
        if input_tensor:
            # Set shape range: min, opt, max
            profile.set_shape(
                name,
                min=(1, 1),              # Minimum batch/sequence
                opt=(batch_size, seq_len),  # Optimal (most common)
                max=(batch_size, seq_len)   # Maximum allowed
            )

    config.add_optimization_profile(profile)

    print("Building TensorRT engine. This may take 5-10 minutes...")
    print("The GPU is optimizing kernels for your specific hardware...")

    # Build serialized network
    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        print("ERROR: Engine build failed.")
        return False

    # Save engine
    engine_data = engine_bytes.serialize() if hasattr(engine_bytes, 'serialize') else bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(engine_data)

    print(f"✓ SUCCESS: Engine saved to {engine_path}")

    # Print engine info
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_data)
        print(f"  Engine size: {len(engine_data) / (1024*1024):.2f} MB")

        # TensorRT 10.x API changes
        if hasattr(engine, 'num_io_tensors'):
            # New API (TensorRT 10.x)
            print(f"  Num IO tensors: {engine.num_io_tensors}")
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                io_type = "Input" if mode == trt.TensorIOMode.INPUT else "Output"
                print(f"    [{io_type}] {name}: {shape}, dtype: {dtype}")

    return True

def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument("--onnx", type=str, default=None,
                        help="Path to ONNX model (default: from config.json)")
    parser.add_argument("--engine", type=str, default=None,
                        help="Path to output engine (default: from config.json)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: from config.json)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length (default: from config.json)")
    parser.add_argument("--config", type=str, default="src/config.json",
                        help="Path to config.json (default: src/config.json)")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 precision instead of FP16")

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
        batch_size = args.batch_size or config.get("batch_size", 256)
        seq_len = args.seq_len or config.get("max_tokens", 32)
        onnx_path = args.onnx or config.get("onnx_path", "onnx/model.onnx")

        # Auto-generate engine path if not specified
        if args.engine:
            engine_path = args.engine
        elif "engine_path" in config:
            engine_path = config["engine_path"]
            # Ensure directory exists
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        else:
            engine_path = "engines/model.engine"
    except FileNotFoundError:
        print(f"Warning: Config file {args.config} not found, using defaults")
        batch_size = args.batch_size or 256
        seq_len = args.seq_len or 32
        onnx_path = args.onnx or "onnx/model.onnx"
        engine_path = args.engine or "engines/model.engine"

    # Check if ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"✗ ERROR: ONNX file not found: {onnx_path}")
        print(f"Please ensure the ONNX model exists at: {os.path.abspath(onnx_path)}")
        return 1

    # Build engine
    use_fp16 = not args.fp32
    if build_engine(onnx_path, engine_path, batch_size, seq_len, use_fp16):
        print(f"\n✓ Engine successfully built!")
        print(f"  Path: {engine_path}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"\nUpdate your config.json with:")
        print(f'  "engine_path": "{engine_path}"')
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())