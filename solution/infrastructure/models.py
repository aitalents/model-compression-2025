from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Dict, Any

import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.nn.utils import prune
import io


@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float


class BaseTextClassificationModel(ABC):

    def __init__(self, name: str, model_path: str, tokenizer: str):
        self.name = name
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.metrics: Dict[str, Any] = {}
        self._load_model()

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def __call__(self, inputs) -> List[TextClassificationModelData]:
        ...


class TransformerTextClassificationModel(BaseTextClassificationModel):

    def _apply_pruning(self, model, amount=0.2):
        """
        Применяем unstructured pruning ко всем слоям Linear модели.
        """
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')  # Удаляем mask, сохраняем pruned веса
        return model

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Применим pruned model
        model = self._apply_pruning(model, amount=0.2)

        # Переводим веса в half precision (FP16), если CUDA доступна
        if self.device == 0:
            model = model.half()
        
        self.model = model.to(self.device)

        # Вычисляем метрики размера
        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
        nonzero_params = total_params - zero_params
        sparsity = zero_params / total_params
        
        # Общий объём весов в памяти
        total_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        # Оценка размера state_dict на диске
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        disk_size_bytes = buffer.getbuffer().nbytes
        buffer.close()

        # Сохраняем в metrics
        self.metrics.update({
            'model_total_parameters': total_params,
            'model_nonzero_parameters': nonzero_params,
            'model_zero_parameters': zero_params,
            'model_sparsity': sparsity,
            'model_memory_size_bytes': total_size_bytes,
            'model_disk_size_bytes': disk_size_bytes
        })

    def tokenize_texts(self, texts: List[str]):
        inputs = self.tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                return_token_type_ids=True,
                return_tensors='pt'
                )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to GPU
        return inputs

    def _results_from_logits(self, logits: torch.Tensor):
        id2label = self.model.config.id2label

        label_ids = logits.argmax(dim=1)
        scores = logits.softmax(dim=-1)
        results = [
                {
                    "label": id2label[label_id.item()],
                    "score": score[label_id.item()].item()
                }
                for label_id, score in zip(label_ids, scores)
            ]
        return results

    def __call__(self, inputs) -> List[TextClassificationModelData]:
        start = time.perf_counter()
        logits = self.model(**inputs).logits
        end = time.perf_counter()
        self.metrics['last_inference_time_sec'] = end - start
        if isinstance(self.device, int):
            self.metrics['last_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        predictions = self._results_from_logits(logits)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return predictions
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

