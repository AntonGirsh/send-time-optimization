import pandas as pd
import numpy as np

# перевод времени в циклическое
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

# добавление фичей, связанных со временем
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour']       = df['send_ts'].dt.hour
    df['dow']        = df['send_ts'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6])
    df['is_morning'] = df['hour'].between(7, 11)
    df['is_lunch']   = df['hour'].between(12, 14)
    df['is_evening'] = df['hour'].between(18, 22)
    df['is_friday']  = df['dow'] == 4
    return df

# построение сетки
def build_time_grid() -> pd.DataFrame:
    grid = []
    for dow in range(7):
        for hour in range(24):
            grid.append({'dow': dow, 'hour': hour, 'day': 15})  # day — любой
    time_grid = pd.DataFrame(grid)
    
    # Добавляем все циклические фичи вручную (без create_time_features!)
    time_grid['hour_sin'] = np.sin(2 * np.pi * time_grid['hour'] / 24)
    time_grid['hour_cos'] = np.cos(2 * np.pi * time_grid['hour'] / 24)
    time_grid['dow_sin'] = np.sin(2 * np.pi * time_grid['dow'] / 7)
    time_grid['dow_cos'] = np.cos(2 * np.pi * time_grid['dow'] / 7)
    time_grid['day_sin'] = np.sin(2 * np.pi * time_grid['day'] / 31)
    time_grid['day_cos'] = np.cos(2 * np.pi * time_grid['day'] / 31)
    
    time_grid['is_weekend'] = (time_grid['dow'] >= 5).astype(int)
    time_grid['hour_bucket'] = pd.cut(time_grid['hour'], bins=[0,6,12,18,24],
                                      labels=['night', 'morning', 'day', 'evening'],
                                      include_lowest=True)
    
    return time_grid