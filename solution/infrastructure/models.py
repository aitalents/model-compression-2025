from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig


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
        self._load_model()

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def __call__(self, inputs) -> List[TextClassificationModelData]:
        ...


class TransformerTextClassificationModel(BaseTextClassificationModel):

    def _load_model(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        #self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model = ORTModelForSequenceClassification.from_pretrained(self.model_path, export=True)
        self.model = self.model.to(self.device)
        self.quantizer = ORTQuantizer.from_pretrained(self.model)
        self.qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
        
        def preprocess_fn(ex, tokenizer):
            return tokenizer(ex["sentence"])

        calibration_dataset = self.quantizer.get_calibration_dataset(
            "glue",
            dataset_config_name="sst2",
            preprocess_function=partial(preprocess_fn, tokenizer=self.tokenizer),
            num_samples=50,
            dataset_split="train",
        )

        # Create the calibration configuration containing the parameters related to calibration.
        calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

        # Perform the calibration step: computes the activations quantization ranges
        ranges = self.quantizer.fit(
            dataset=calibration_dataset,
            calibration_config=calibration_config,
            operators_to_quantize=self.qconfig.operators_to_quantize,
        )

        # Apply static quantization on the model
        model_quantized_path = self.quantizer.quantize(
            save_dir="infrastructure",
            calibration_tensors_range=ranges,
            quantization_config=self.qconfig,
        )

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
        logits = self.model(**inputs).logits
        predictions = self._results_from_logits(logits)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return predictions