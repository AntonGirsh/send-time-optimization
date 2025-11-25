import pandas as pd
import numpy as np
from typing import List

def create_time_features(df: pd.DataFrame, ts_column: str = 'send_ts') -> pd.DataFrame:
    """
    Вход:
        df — DataFrame с колонкой send_ts (datetime)
    
    Выход:
        df с новыми колонками: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
    """
    print()
    print('Преобразование времени в цикличное:')
    
    df = df.copy()
    ts = pd.to_datetime(df[ts_column])
    
    df['hour'] = ts.dt.hour
    df['dow']  = ts.dt.dayofweek          # 0=понедельник, 6=воскресенье
    df['day']  = ts.dt.day                # 1–31
    
    # === Циклические преобразования (критически важно!) ===
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['dow']  / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['dow']  / 7)
    df['day_sin']  = np.sin(2 * np.pi * df['day']  / 31)
    df['day_cos']  = np.cos(2 * np.pi * df['day']  / 31)
    
    # === Бизнес-флаги (подстраиваются под РФ/СНГ) ===
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    print('Done')
    return df


def build_time_grid() -> pd.DataFrame:
    """
    Создаёт сетку всех возможных часов недели (168 строк).
    """
    hours = list(range(24))
    dows = list(range(7))
    grid = []

    for dow in dows:
        for hour in hours:
            grid.append({'dow': dow, 'hour': hour})

    time_grid = pd.DataFrame(grid)
    
    # Применяем те же фичи, что и в create_time_features
    temp_df = pd.DataFrame({'send_ts': pd.to_datetime('2025-01-01')})
    temp_df = create_time_features(temp_df, 'send_ts')
    
    # Копируем логику для каждой строки
    time_grid['hour_sin'] = np.sin(2 * np.pi * time_grid['hour'] / 24)
    time_grid['hour_cos'] = np.cos(2 * np.pi * time_grid['hour'] / 24)
    time_grid['dow_sin'] = np.sin(2 * np.pi * time_grid['dow'] / 7)
    time_grid['dow_cos'] = np.cos(2 * np.pi * time_grid['dow'] / 7)
    time_grid['day'] = 15  # фиктивное значение, не используется
    time_grid['day_sin'] = np.sin(2 * np.pi * time_grid['day'] / 31)
    time_grid['day_cos'] = np.cos(2 * np.pi * time_grid['day'] / 31)


    time_grid['is_weekend'] = time_grid['dow'].isin([5, 6]).astype(int)
    return time_grid