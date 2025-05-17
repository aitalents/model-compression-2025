import os
import time
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import pandas as pd
import psutil
import torch
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor


def get_memory_usage():
    """Get RAM usage in MB"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    return ram_usage

def get_gpu_memory_usage():
    """Get VRAM usage in MB"""
    torch.cuda.synchronize()
    vram_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    return vram_usage


def load_model_and_dataset(device="cpu"):
    """Load ViT model and beans dataset"""
    print("Loading model and dataset...")
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    dataset = load_dataset("beans", download_mode="force_redownload", verification_mode='no_checks')
    
    def transform(examples):
        return processor(examples["image"], return_tensors="pt").to(device)
    
    processed_dataset = dataset["test"].select(range(32))
    processed_inputs = transform(processed_dataset)
    
    labels = processed_dataset["labels"]
    
    return model, processed_inputs, labels, processor


def export_to_openvino(model, device="cpu", output_path="hw5/openvino_model"):
    """Export model to OpenVINO format"""
    print("Exporting model to OpenVINO...")
    model.eval()
    
    dummy_input = torch.ones((1, 3, 224, 224), dtype=torch.float32).to(device)
    
    onnx_path = "hw5/temp_model.onnx"
    print(f"Exporting to ONNX first (temporary file: {onnx_path})...")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=14,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
            do_constant_folding=True
        )
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checked - OK")
    
    print(f"Converting ONNX to OpenVINO IR (output: {output_path})...")
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, output_path + ".xml")
    
    os.remove(onnx_path)
    print(f"Model exported to {output_path}.xml")
    
    return output_path + ".xml"


def measure_performance_openvino(model_path, inputs, labels, device="cpu"):
    """Measure OpenVINO model performance"""
    print(f"Measuring performance for OpenVINO model on {device}...")

    core = ov.Core()
    available_devices = core.available_devices
    print(f"Available OpenVINO devices: {available_devices}")
    
    if device == "cuda":
        ov_device = "GPU"
    else:
        ov_device = "CPU"
    
    print(f"Loading OpenVINO model on {ov_device}...")
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model, ov_device)
    infer_request = compiled_model.create_infer_request()
    input_tensor = inputs["pixel_values"].cpu().numpy()
    ram_usage = get_memory_usage()
    
    if device == "cuda":
        vram_usage = get_gpu_memory_usage()
    else:
        vram_usage = 0
    for _ in range(10):
        infer_request.infer(inputs={"pixel_values": input_tensor})
    
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        result = infer_request.infer(inputs={"pixel_values": input_tensor})
        outputs = next(iter(result.values()))
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / 100  # Average time in ms
    
    predictions = np.argmax(outputs, axis=1)
    accuracy = (predictions == labels).mean() * 100
    
    model_size_mb = (os.path.getsize(model_path) + os.path.getsize(model_path.replace(".xml", ".bin"))) / (1024 * 1024)
    
    return {
        "Модель": "google/vit-base-patch16-224-in21k",
        "Метод": "OpenVINO",
        "Размер весов": f"{model_size_mb:.2f}Mb",
        "Время инференса (CPU, ms)": f"{avg_time_ms:.2f}ms",
        "Использование RAM (MB)": f"{ram_usage:.2f}Mb",
        "Качество (Accuracy)": f"{accuracy:.2f}%",
    }


def measure_performance(model, inputs, labels, device, export_type="original"):
    """Measure original model performance"""
    print(f"Measuring performance for {export_type} model on {device}...")
    
    model = model.to(device)
    model.eval()
    
    input_tensor = inputs["pixel_values"].to(device)
    
    ram_usage = get_memory_usage()
    
    if device == "cuda":
        vram_usage = get_gpu_memory_usage()
    else:
        vram_usage = 0
    
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / 100  # Average time in ms
    
    predictions = outputs.logits.argmax(-1).cpu().numpy()
    accuracy = (predictions == labels).mean() * 100
    
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        "Модель": "google/vit-base-patch16-224-in21k",
        "Метод": export_type,
        "Размер весов": f"{model_size_mb:.2f}Mb",
        "Время инференса (CPU, ms)": f"{avg_time_ms:.2f}ms",
        "Использование RAM (MB)": f"{ram_usage:.2f}Mb",
        "Качество (Accuracy)": f"{accuracy:.2f}%",
    }



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("No GPU available! This script requires GPU.")
        return
    
    os.makedirs("hw5", exist_ok=True)
    
    model, inputs, labels, processor = load_model_and_dataset(device)
    
    original_perf = measure_performance(model, inputs, labels, device, "Оригинал")
    
    openvino_model_path = export_to_openvino(model, processor, device)
    
    openvino_perf = measure_performance_openvino(openvino_model_path, inputs, labels, device)
    
    results = pd.DataFrame([original_perf, openvino_perf])
    print("\nРезультаты сравнения:")
    print(results.to_markdown(index=False))

if __name__ == "__main__":
    main() 