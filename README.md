# Store review rating classification

5-классовая классификация отзывов.

## Что внутри

- `train_transformer_ordinal.py` — основная модель: ordinal classification поверх RuBERT с кросс-валидацией.
- `train_linear_cv.py` — быстрый TF-IDF бэкап на word/char n-gram.
- `build_submission.py` — сборка финального submission из вероятностей моделей.
- `requirements.txt` — зависимости.

## Идея

Целевая переменная упорядочена: 1 < 2 < 3 < 4 < 5. Поэтому основная модель обучается не как обычный flat multiclass, а как ordinal classification по 4 порогам:

- `y > 1`
- `y > 2`
- `y > 3`
- `y > 4`

Это обычно устойчивее на соседних классах и лучше отражает природу рейтинга.

Дополнительно есть быстрый линейный бэкап:

- word TF-IDF `(1, 2)`
- char TF-IDF `(3, 5)`
- усреднение вероятностей двух веток

## Как запускать

### 1. Установка

```bash
pip install -r requirements.txt
```

### 2. Основная модель

```bash
python train_transformer_ordinal.py \
  --train train.csv \
  --test test.csv \
  --output-dir runs/rubert_ordinal \
  --model-name DeepPavlov/rubert-base-cased-conversational \
  --folds 5 \
  --epochs 4 \
  --batch-size 16 \
  --grad-accum 2 \
  --max-length 192 \
  --lr 2e-5
```

Если памяти мало, можно заменить backbone:

```bash
--model-name cointegrated/rubert-tiny2
```

### 3. Быстрый линейный бэкап

```bash
python train_linear_cv.py \
  --train train.csv \
  --test test.csv \
  --output-dir runs/linear_tfidf \
  --folds 5
```

### 4. Сборка submission

С ансамблем:

```bash
python build_submission.py \
  --sample sample_submission.csv \
  --transformer-probs runs/rubert_ordinal/test_probs.npy \
  --linear-probs runs/linear_tfidf/test_probs.npy \
  --transformer-weight 0.85 \
  --linear-weight 0.15 \
  --out submission.csv
```

Только трансформер:

```bash
python build_submission.py \
  --sample sample_submission.csv \
  --transformer-probs runs/rubert_ordinal/test_probs.npy \
  --out submission.csv
```

## Что смотреть после обучения

У каждой модели сохраняются:

- `metrics.json`
- `oof_probs.npy`
- `test_probs.npy`

Главная цифра — `oof_weighted_f1`.

## Практические замечания

- Для этой задачи лучше не брать слишком большой `max_length`: отзывы в среднем короткие.
- Если есть 12+ GB VRAM, можно поднять `batch-size`.
- Если модель начала переобучаться, сначала уменьшай `epochs`, а не усложняй код.
- Если хочется выжать ещё немного, полезно прогнать 2 сидa и усреднить вероятности между запусками.
