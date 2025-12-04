# Send-Time Optimization. Модель для выбора оптимального времени отправки офферов

## Установка

pip install -r requirements.txt

## Генерация данных

python run.py data --n 5000

## Обучение

python run.py train data/synthetic.parquet --run-id 2025-12-01_v1

## Инференс

python run.py predict 2025-12-01_v1 data/synthetic.parquet --output reports/pred.parquet

## Визуализация

python run.py viz uplift-heatmap reports/pred.parquet

## Интерпретация

python run.py interp profile-top-slots -p reports/pred.parquet --min-uplift 0.03 --min-clients-abs 100

## feature_importance

python run.py f show 2025-12-01_v2 --top 15
