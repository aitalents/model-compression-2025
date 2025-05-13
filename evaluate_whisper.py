#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate original Whisper-large-v3 vs. INT8-quantised ONNX export.

Metrics:
    • weight size on disk (MB)
    • mean inference time  (CPU / GPU)  – seconds per text sample
    • peak RAM and VRAM during inference (MB)
    • word-error-rate (WER, lower is better)
"""

import os, sys, time, shutil, json, gc, contextlib
from pathlib import Path
import psutil
import torch
from tqdm import tqdm
from jiwer import wer
from datasets import load_dataset, Audio
from transformers import AutoProcessor, pipeline, AutoModelForSpeechSeq2Seq
from transformers.utils import logging as hf_logging
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import pandas as pd
import pynvml
from huggingface_hub import snapshot_download

hf_logging.set_verbosity_error()  # cleaner console

# ---------- CONFIG ----------------------------------------------------------------
original_model_name = "openai/whisper-large-v3"
# локальная папка с *.onnx
export_dir_int8 = "whisper_onnx_int8"
# Сколько аудио оценивать (уменьшено ради скорости на CPU)
N_SAMPLES = 20
# We switch to a much lighter dataset – Common Voice (Russian test split)
# It streams the data and only downloads the exact number of audio clips we select.
# Feel free to change language or split if desired.
DS_NAME = "mozilla-foundation/common_voice_13_0"
# language code ("ru" keeps consistency with your Russian example)
DS_CONFIG = "ru"
DS_SPLIT = "test"
SAMPLE_RATE = 16000                          # Whisper expects 16 kHz
# ----------------------------------------------------------------------------------

def sizeof_dir(path: Path) -> int:
    """Directory size in MB (binary MiB)."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total // (1024 ** 2)

@contextlib.contextmanager
def track_memory():
    """Context manager → yields (ram_mb_before, vram_mb_before)."""
    process = psutil.Process(os.getpid())
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def mem():
        ram = process.memory_info().rss // (1024 ** 2)
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used // (1024 ** 2)
        return ram, vram

    start_ram, start_vram = mem()
    peak_ram, peak_vram = start_ram, start_vram
    try:
        yield lambda: (peak_ram, peak_vram)
    finally:
        peak_ram, peak_vram = mem()  # last snapshot
        track_memory.peak_ram = max(track_memory.peak_ram, peak_ram) if hasattr(track_memory, "peak_ram") else peak_ram  # noqa
        track_memory.peak_vram = max(track_memory.peak_vram, peak_vram) if hasattr(track_memory, "peak_vram") else peak_vram  # noqa
    pynvml.nvmlShutdown()

def run_inference(pipe, dataset, device):
    preds, refs = [], []
    def to_pipe_input(audio_field):
        """Convert dataset's audio column to the format expected by ASR pipeline."""
        # HF datasets Audio feature returns dict with 'array' and 'sampling_rate'
        if isinstance(audio_field, dict):
            return {"raw": audio_field["array"], "sampling_rate": audio_field["sampling_rate"]}
        # Fallback: object with attributes
        return {"raw": audio_field.array, "sampling_rate": audio_field.sampling_rate}

    with torch.inference_mode(), torch.no_grad():
        _ = pipe(to_pipe_input(dataset[0]["audio"]), batch_size=1)

    # --- основное измерение --------------------------------------------------------
    start = time.perf_counter()
    with torch.inference_mode(), torch.no_grad():
        for sample in tqdm(dataset, desc=f"Inference-{device}", leave=False):
            audio_input = to_pipe_input(sample["audio"])
            out = pipe(audio_input, batch_size=1)
            preds.append(out["text"].strip().lower())
            # Common Voice uses "sentence" instead of "text"; fallback accordingly
            ref = sample.get("text") or sample.get("sentence") or ""
            refs.append(ref.strip().lower())
    total = time.perf_counter() - start
    return total / len(dataset), wer(refs, preds)

def prepare_dataset():
    """Load a lightweight streaming dataset and return an iterable with exactly N_SAMPLES examples."""
    ds = load_dataset(DS_NAME, DS_CONFIG, split=DS_SPLIT, streaming=True, trust_remote_code=True)
    # Common Voice column that holds reference transcript is "sentence"
    # We cast audio on-the-fly to TARGET sample rate.
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # Take first N_SAMPLES examples deterministically (streaming datasets support .take())
    if N_SAMPLES:
        ds = ds.take(N_SAMPLES)
    return list(ds)  # materialise to list so we can iterate multiple times if needed

def load_pipe_hf(model_name, device_str):
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch.float16 if "cuda" in device_str else torch.float32).to(device_str)
    return pipeline("automatic-speech-recognition",
                    model=model, tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
                    device=0 if "cuda" in device_str else -1,
                    chunk_length_s=30, batch_size=1)

def load_pipe_onnx(onnx_dir, device_str):
    proc = AutoProcessor.from_pretrained(original_model_name)
    providers = ["CUDAExecutionProvider"] if "cuda" in device_str else ["CPUExecutionProvider"]
    model = ORTModelForSpeechSeq2Seq.from_pretrained(onnx_dir, provider=providers[0])
    return pipeline("automatic-speech-recognition",
                    model=model, tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
                    device=0 if "cuda" in device_str else -1,
                    chunk_length_s=30, batch_size=1)

def evaluate(model_label, loader_fn, path_or_name):
    metrics = {"Model": model_label[0], "Method": model_label[1]}
    ds = prepare_dataset()

    # weight size
    if Path(path_or_name).exists():
        local_path = Path(path_or_name)
    else:
        local_path = Path(
            snapshot_download(
                repo_id=path_or_name,
                allow_patterns=["*.bin", "*.safetensors", "*.onnx"],
                local_files_only=False,
            )
        )
    metrics["Размер весов"] = f"{sizeof_dir(local_path)} МB"

    for dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            metrics["Время инференса (GPU)"] = "-"
            metrics["Использование VRAM (MB)"] = "-"
            continue

        pipe = loader_fn(path_or_name, dev)
        with track_memory() as mem_func:
            sec_per_txt, wer_val = run_inference(pipe, ds, dev)
            peak_ram, peak_vram = mem_func()
        label_time = "Время инференса (GPU)" if dev == "cuda" else "Время инференса (CPU)"
        label_ram = "Использование VRAM (MB)" if dev == "cuda" else "Использование RAM (MB)"
        metrics[label_time] = f"{sec_per_txt:.3f}"
        metrics[label_ram] = f"{peak_vram if dev=='cuda' else peak_ram} MB"
        if dev == "cuda":
            metrics["Точность (WER)"] = f"{wer_val:.3f}"

        # cleanup
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return metrics


def main():
    results = []
    results.append(evaluate(("Whisper-large-v3", "Original"), load_pipe_hf, original_model_name))
    results.append(evaluate(("Whisper-large-v3", "ONNX-int8"), load_pipe_onnx, export_dir_int8))

    df = pd.DataFrame(results)
    order = ["Model", "Method", "Размер весов",
             "Время инференса (CPU)", "Время инференса (GPU)",
             "Использование RAM (MB)", "Использование VRAM (MB)", "Точность (WER)"]
    df = df[order]
    print("\n" + df.to_markdown(index=False, tablefmt="github"))


if __name__ == "__main__":
    main()