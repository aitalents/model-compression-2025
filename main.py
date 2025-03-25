import os
import time
import json
import psutil
import argparse

import torch
import numpy as np

import evaluate
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME_OR_PATH = "google/vit-base-patch16-224-in21k"
DATASET_NAME = "beans"


def get_ram_usage_mb():
    return psutil.Process().memory_info().rss / 1024**2


def get_vram_usage_mb():
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0


def load_model_and_processor(model_name: str, device: torch.device):
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def preprocess_input(example, processor, device: torch.device):
    return processor(example["image"], return_tensors="pt").to(device)


def get_dir_size_mb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024**2


def evaluate_model(model, dataset, processor, device: torch.device):
    predictions, references = [], []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_ram = get_ram_usage_mb()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for example in dataset:
        inputs = preprocess_input(example, processor, device)
        references.append(example["labels"])

        with torch.no_grad():
            output = model(**inputs)

        predictions.append(output.logits.argmax(-1).item())

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    total_inference_time = (end_time - start_time) * 1000

    end_ram = get_ram_usage_mb()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0

    return {
        f"{device.type}_inference_time_ms": total_inference_time,
        "peak_ram_usage": end_ram - start_ram,
        "peak_vram_usage": peak_vram,
        "predictions": predictions,
        "references": references
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = load_dataset(DATASET_NAME, split="test")
    model, processor = load_model_and_processor(MODEL_NAME_OR_PATH, device)
    metric = evaluate.load("accuracy")

    metrics = evaluate_model(model, dataset, processor, device)
    accuracy = metric.compute(predictions=metrics["predictions"], references=metrics["references"])
    metrics["accuracy"] = accuracy["accuracy"]

    metrics.pop("predictions")
    metrics.pop("references")
    
    # for validation
    model.save_pretrained("vit_beans")
    size_mb = get_dir_size_mb("vit_beans")
    metrics["weights_size"] = size_mb

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
