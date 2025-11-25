import pandas as pd
import numpy as np

def create_time_features(df: pd.DataFrame, ts_column: str = 'send_ts') -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_column])
    
    df['hour'] = ts.dt.hour
    df['dow'] = ts.dt.dayofweek
    df['day'] = ts.dt.day
    
    # Циклические
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Флаги
    df['is_weekend'] = (df['dow'] >= 5).astype(int)    

    return df


def build_time_grid() -> pd.DataFrame:
    grid = []
    for dow in range(7):
        for hour in range(24):
            grid.append({'dow': dow, 'hour': hour, 'day': 15})
    time_grid = pd.DataFrame(grid)
    
    # Применяем фичи
    time_grid = create_time_features(pd.DataFrame({'send_ts': pd.to_datetime('2025-01-01')}), 'send_ts')
    time_grid['send_ts'] = pd.to_datetime('2025-01-01')  # заглушка
    
    # Применяем к каждой строке
    time_grid['hour_sin'] = np.sin(2 * np.pi * time_grid['hour'] / 24)
    # ... (аналогично для всех фичей из create_time_features)
    
    return time_grid