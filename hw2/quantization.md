
# Сравнительная таблица

| Модель    |         Метод         | Размер весов | Время инференса (CPU, ms) | Время инференса (GPU, ms) | Использование RAM (MB) | Использование VRAM (MB) |  Качество (Precision, Recall, F1-Score))  |
| :-------------- | :-------------------------: | :---------------------: | :-------------------------------------: | :-------------------------------------: | :---------------------------------: | :----------------------------------: | :-----------------------------------------------: |
| efficientvit_b3 |      Оригинал      |        185.75 Mb        |               113.27 ms               |                21.26 ms                |              764.75 MB              |              539.00 Mb              | 0.8474 precision 0.8342 recall 0.8239 f1-score |
| efficientvit_b3 | Dynamic quantization (fp16) |        153.48 MB        |                35.89 ms                |              not supported              |              841.46 MB              |            not supported            |  0.8474 precision 0.8342 recall 0.8239 f1-score  |
| efficientvit_b3 | Dynamic quantization (int8) |        153.48 MB        |                33.11 ms                |              not supported              |              829.30 MB              |            not supported            | 0.8482 precision 0.8332 recall 0.8236 f1-score |
