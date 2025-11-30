# src/data_generation.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.features import add_time_features

def generate_and_save(n_samples: int = 5000, path: Path = Path("data/synthetic.parquet")) -> None:
    np.random.seed(42)

    # 1. Равномерно распределённые случайные даты в течение года
    start = pd.Timestamp('2025-01-01')
    end   = pd.Timestamp('2025-12-31 23:59')
    random_timestamps = pd.date_range('2025-01-01', '2025-12-31 23:59', periods=n_samples).\
            to_series().sample(n_samples, replace=True).sort_values().reset_index(drop=True)
    
    df = pd.DataFrame({
        'client_id': range(1, n_samples + 1),
        'send_ts': random_timestamps,
    })

    # 2. добавляем фичи времени
    df = add_time_features(df)

    # 3. Клиентские признаки
    df['age']            = np.random.randint(18, 70, n_samples)
    df['app_intensity']  = np.random.uniform(0.1, 10.0, n_samples).round(2)
    df['country']        = np.random.choice(['RU', 'KZ', 'BY', 'UA'], n_samples, p=[0.4, 0.15, 0.2, 0.25])

    # 4. Базовые вероятности
    df['p_bank'] = 0.7
    df['p_user'] = 0.6

    # ГРУППЫ С ЯВНЫМИ ВРЕМЕННЫМИ ПРЕДПОЧТЕНИЯМИ (каждая строчка — отдельная группа!)

    # Молодые активные — очень любят пятницу вечером
    df['p_bank'] += (df['age'] < 35) & (df['app_intensity'] > 7.0) & (df['is_friday'] & df['is_evening']) * 0.10
    df['p_user'] += (df['age'] < 35) & (df['app_intensity'] > 7.0) & (df['is_friday'] & df['is_evening']) * 0.10

    # Люди 40+ — уважают утро будних дней
    df['p_bank'] += (df['age'] >= 40) & (df['is_monday'] | df['dow'].between(1, 3)) & df['is_morning'] * 0.12
    df['p_user'] += (df['age'] >= 40) & (df['is_monday'] | df['dow'].between(1, 3)) & df['is_morning'] * 0.12

    # Все ненавидят обед (особенно в будни)
    df['p_bank'] -= df['is_lunch'] & ~df['is_weekend'] * 0.2
    df['p_user'] -= df['is_lunch'] & ~df['is_weekend'] * 0.2

    # Выходные утром — провал (кроме редких любителей)
    df['p_bank'] -= df['is_weekend'] & df['is_morning'] * 0.1
    df['p_user'] -= df['is_weekend'] & df['is_morning'] * 0.1

    # Лёгкий вечерний буст для всех
    df['p_user'] += df['is_evening'] & ~df['is_friday'] * 0.07

    # 5. Клиппим, чтобы не было 0 и 1
    df['p_bank'] = df['p_bank'].clip(0.05, 0.98)
    df['p_user'] = df['p_user'].clip(0.02, 0.95)

    # 6. Финальные бинарные таргеты
    df['accept_bank'] = np.random.binomial(1, df['p_bank'])
    df['accept_user'] = df['accept_bank'] * np.random.binomial(1, df['p_user'])

    # 7. Оставляем только нужные колонки
    result = df[[
        'client_id', 'send_ts',
        'age', 'app_intensity', 'country',
        'accept_bank', 'accept_user'
    ]].copy()

    path.parent.mkdir(exist_ok=True)
    result.to_parquet(path, index=False)