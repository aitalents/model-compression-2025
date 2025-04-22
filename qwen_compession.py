import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig, AwqConfig
from datasets import load_dataset
import evaluate
import time
import psutil
import os
import pandas as pd
import shutil
import torch.nn.utils.prune as prune
import gc

MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
N_SAMPLES_EVAL = 100
N_SAMPLES_CALIBRATION = 128
INFERENCE_TEXT = "Hello, how are you?"
MAX_NEW_TOKENS_EVAL = 50
MAX_NEW_TOKENS_INFERENCE = 50
PRUNING_AMOUNT = 0.3


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared GPU Memory and ran GC.")


def measure_inference_time(model, tokenizer, text, device, max_new_tokens=MAX_NEW_TOKENS_INFERENCE):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    end_time = time.time()
    return (end_time - start_time) * 1000  # in milliseconds


def measure_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


def measure_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    else:
        return 0


def measure_disk_size(model, tokenizer, temp_dir="temp_model_size"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    print(f"Saving model to {temp_dir} to measure disk size...")
    try:
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(temp_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        print(f"Done saving and measuring size.")
        return total_size / (1024 ** 2)  # in MB
    except Exception as e:
        print(f"Warning: Could not save model to measure disk size: {e}")
        return -1
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def evaluate_quality(model, tokenizer, texts, device, max_new_tokens=MAX_NEW_TOKENS_EVAL):
    model.eval()
    bleu_metric = evaluate.load("bleu")
    references = []
    predictions = []
    texts = [s for s in texts if s and s.strip()]

    print(f"Evaluating BLEU on {len(texts)} samples...")
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        references.append([text])
        predictions.append(prediction)

    if not predictions:
        print("Warning: No predictions generated for BLEU evaluation.")
        return 0.0

    valid_predictions = [p if p else "" for p in predictions]
    valid_references = [[r[0] if r and r[0] else ""] for r in references]

    min_len = min(len(valid_predictions), len(valid_references))
    if min_len < len(predictions):
        print(
            f"Warning: Length mismatch or filtering reduced samples for BLEU from {len(predictions)} to {min_len}")

    if min_len == 0:
        print("Warning: No valid prediction/reference pairs for BLEU computation.")
        return 0.0

    results = bleu_metric.compute(predictions=valid_predictions[:min_len], references=valid_references[:min_len])
    print(f"BLEU score computed: {results['bleu']:.4f}")
    return results["bleu"]


def get_calibration_dataset(dataset_name, config_name, split, n_samples, tokenizer, max_length=512):
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=False)
    texts = []
    count = 0
    for example in dataset:
        if count >= n_samples:
            break
        text = example.get("text")
        if text and text.strip():
            texts.append(text.strip())
            count += 1

    print(f"Calibration dataset loaded with {len(texts)} samples.")
    return texts


def load_evaluation_texts():
    eval_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, streaming=False)
    eval_texts_raw = [ex.get("text") for ex in eval_dataset]
    eval_texts = [s for s in eval_texts_raw if s and s.strip()][:N_SAMPLES_EVAL]
    return eval_texts


def run_fp32_baseline(eval_texts, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ram_before_load = measure_ram_usage()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    ram_after_load = measure_ram_usage()
    model_size_disk = measure_disk_size(model, tokenizer)
    model.to(device)
    clear_gpu_memory()
    vram_usage_gpu = measure_vram_usage()
    gpu_time = measure_inference_time(model, tokenizer, INFERENCE_TEXT, device)
    quality_bleu_gpu = evaluate_quality(model, tokenizer, eval_texts, device)
    result = {
        "Technique": "FP32 (Baseline)",
        "Inference Time (GPU, ms)": gpu_time,
        "VRAM Usage (MB)": vram_usage_gpu,
        "Quality (BLEU GPU)": quality_bleu_gpu
    }
    del model, tokenizer
    clear_gpu_memory()
    return result


def run_fp16(eval_texts, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    clear_gpu_memory()
    vram_usage = measure_vram_usage()
    inference_time = measure_inference_time(model, tokenizer, INFERENCE_TEXT, device)
    bleu_score = evaluate_quality(model, tokenizer, eval_texts, device)
    disk_size = measure_disk_size(model, tokenizer)
    del model, tokenizer
    clear_gpu_memory()
    return {
        "Technique": "FP16",
        "Inference Time (GPU, ms)": inference_time,
        "VRAM Usage (MB)": vram_usage,
        "Quality (BLEU GPU)": bleu_score
    }


def run_int8(eval_texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant_config, device_map="auto", low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    clear_gpu_memory()
    primary_device = model.device
    vram_usage = measure_vram_usage()
    inference_time = measure_inference_time(model, tokenizer, INFERENCE_TEXT, primary_device)
    bleu_score = evaluate_quality(model, tokenizer, eval_texts, primary_device)
    del model, tokenizer
    clear_gpu_memory()
    return {
        "Technique": "INT8 (BitsAndBytes)",
        "Inference Time (GPU, ms)": inference_time,
        "VRAM Usage (MB)": vram_usage,
        "Quality (BLEU GPU)": bleu_score
    }


def run_gptq(eval_texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    calibration_texts = get_calibration_dataset(DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, N_SAMPLES_CALIBRATION, tokenizer)
    gptq_config = GPTQConfig(bits=4, dataset=calibration_texts, tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=gptq_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    clear_gpu_memory()
    primary_device = model.device
    vram_usage = measure_vram_usage()
    inference_time = measure_inference_time(model, tokenizer, INFERENCE_TEXT, primary_device)
    bleu_score = evaluate_quality(model, tokenizer, eval_texts, primary_device)
    del model, tokenizer
    clear_gpu_memory()
    return {
        "Technique": "GPTQ (4-bit)",
        "Inference Time (GPU, ms)": inference_time,
        "VRAM Usage (MB)": vram_usage,
        "Quality (BLEU GPU)": bleu_score
    }


def run_awq(eval_texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    awq_config = AwqConfig(bits=4)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat-AWQ",
        quantization_config=awq_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    clear_gpu_memory()
    primary_device = model.device
    vram_usage = measure_vram_usage()
    inference_time = measure_inference_time(model, tokenizer, INFERENCE_TEXT, primary_device)
    bleu_score = evaluate_quality(model, tokenizer, eval_texts, primary_device)
    del model, tokenizer
    clear_gpu_memory()
    return {
        "Technique": "AWQ (4-bit)",
        "Inference Time (GPU, ms)": inference_time,
        "VRAM Usage (MB)": vram_usage,
        "Quality (BLEU GPU)": bleu_score
    }


def run_pruning(eval_texts, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    if parameters_to_prune:
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=PRUNING_AMOUNT)
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
    model_size = measure_disk_size(model, tokenizer)
    model.to(device)
    clear_gpu_memory()
    vram_usage = measure_vram_usage()
    inference_time = measure_inference_time(model, tokenizer, INFERENCE_TEXT, device)
    bleu_score = evaluate_quality(model, tokenizer, eval_texts, device)
    del model, tokenizer
    clear_gpu_memory()
    return {
        "Technique": f"Pruning ({PRUNING_AMOUNT * 100:.0f}%)",
        "Inference Time (GPU, ms)": inference_time,
        "VRAM Usage (MB)": vram_usage,
        "Quality (BLEU GPU)": bleu_score
    }


def main():
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_texts = load_evaluation_texts()
    results.append(run_fp32_baseline(eval_texts, device))
    results.append(run_fp16(eval_texts, device))
    results.append(run_int8(eval_texts))
    results.append(run_gptq(eval_texts))
    results.append(run_awq(eval_texts))
    results.append(run_pruning(eval_texts, device))
    df = pd.DataFrame(results)
    print(df)


if __name__ == '__main__':
    main()

