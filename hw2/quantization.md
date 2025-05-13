
# Сравнительная таблица

| Модель          | Метод                       | Размер весов | Время инференса (CPU, ms) | Время инференса (GPU, ms) | Использование RAM (MB) | Использование VRAM (MB) | Качество (Precision, Recall, F1-Score))          |
|-----------------|-----------------------------|--------------|---------------------------|---------------------------|------------------------|-------------------------|--------------------------------------------------|
| efficientvit_b3 | Оригинал                    | 185.75 Mb    | 113.27 ms                 | 21.26 ms                  | 764.75 MB              | 539.00 Mb               | 0.8474 precision 0.8342 recall 0.8239 f1-score   |
| efficientvit_b3 | Dynamic quantization (fp16) | 153.48 MB    | 35.89 ms                  | not supported             | 841.46 MB              | not supported           | 0.8474 precision 0.8342 recall 0.8239 f1-score   |
| efficientvit_b3 | Dynamic quantization (int8) | 153.48 MB    | 33.11 ms                  | not supported             | 829.30 MB              | not supported           | 0.8482 precision 0.8332 recall 0.8236 f1-score   |
| efficientvit_b3 | Unstructed pruning          | 338.65 MB    | 123.89 ms                 | 23.26 ms                  | 940.97 MB              | 599.02 MB               | 0.7931 precision 0.7602 recall 0.7479 f1-score   |
| efficientvit_b3 | Structed pruning            | 118.55 MB    | 110.98 ms                 | 20.14 ms                  | 690.84 MB              | 510.23 MB               | 0.6626 precision 0.6210 recall 0.6061 f1-score   |