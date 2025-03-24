import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import time
import psutil
import os
import pandas as pd

# Функция для измерения времени инференса
def measure_inference_time(model, tokenizer, text, device='cpu'):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)  # Генерация текста
    end_time = time.time()
    return (end_time - start_time) * 1000  # в миллисекундах

# Функция для измерения использования памяти
def measure_memory_usage(device_type="cpu"):
    process = psutil.Process(os.getpid())
    if device_type == "cpu":
        return process.memory_info().rss / 1024 ** 2  # в MB
    elif device_type == "gpu":
        return torch.cuda.memory_allocated() / 1024 ** 2  # в MB

# Функция для измерения размера модели
def measure_model_size(model):
    param_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2  # в MB
    return param_size

# Функция для оценки качества модели с использованием метрики BLEU
def evaluate_quality(model, tokenizer, texts, device='cpu'):
    bleu = evaluate.load("bleu")
    references = []
    predictions = []
    texts = [s for s in texts if s!='']
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        references.append([text])
        predictions.append(prediction)
    results = bleu.compute(predictions=predictions, references=references)
    return results["bleu"]

if __name__ == '__main__':
    
    # Загрузка модели и токенизатора
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Загрузка датасета
    
    # Берутся 100 примеров из-за дефицита ресурсов
    butch_texts = load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"][:100]
    text = "Hello, how are you?" # Текст для замера скорости инференса
    
    # Измерение характеристик на cpu
    cpu_time = measure_inference_time(model, tokenizer, text, torch.device("cpu"))
    ram_usage_cpu = measure_memory_usage("cpu")
    quality_bleu_cpu = evaluate_quality(model, tokenizer, butch_texts, torch.device("cpu"))
    
    # Измерение характеристик на gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gpu_time = measure_inference_time(model, tokenizer, text, torch.device("cuda"))
    vram_usage_gpu = measure_memory_usage("gpu")
    quality_bleu_gpu = evaluate_quality(model, tokenizer, butch_texts, torch.device("cuda"))
    
    # Измерение размера модели
    model_size = measure_model_size(model)

    results_table = pd.DataFrame({
        "Название модели": [model_name],
        "Размер весов (MB)": [model_size],
        "Время инференса (CPU, ms)": [cpu_time],
        "Время инференса (GPU, ms)": [gpu_time],
        "Использование RAM (MB)": [ram_usage_cpu],
        "Использование VRAM (MB)": [vram_usage_gpu],
        "Качество (BLEU) на CPU": [quality_bleu_cpu],
        "Качество (BLEU) на GPU": [quality_bleu_gpu]
    })
    #print(results_table)
    results_table.to_csv("result.csv", index=False)