import pandas as pd
import numpy as np
from src.metrics import harmonic_f1

def predict_best_time_for_dataset(
    df: pd.DataFrame,
    time_grid: pd.DataFrame,
    bank_model,
    user_model,
    bank_calibrator,
    user_calibrator,
    cfg,
    top_k: int = 3
) -> pd.DataFrame:
    cat_features = [f for f in cfg.features.categorical if f in df.columns]
    num_features = [f for f in cfg.features.numeric if f in df.columns]
    time_features = [f for f in cfg.features.time.all_generated if f in df.columns]
    feature_cols = cat_features + num_features + time_features

    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype('string').fillna('missing')

    results = []
    for _, client_row in df.iterrows():
        candidates = pd.DataFrame([client_row.to_dict()] * len(time_grid))
        for col in time_grid.columns:
            if col in feature_cols:
                candidates[col] = time_grid[col].values

        for col in cat_features:
            if col in candidates.columns:
                candidates[col] = candidates[col].astype('string').fillna('missing')

        X = candidates[feature_cols]

        p_bank_raw = bank_model.predict_proba(X)[:, 1]
        p_user_raw = user_model.predict_proba(X)[:, 1]
        p_bank = bank_calibrator.predict(p_bank_raw)
        p_user = user_calibrator.predict(p_user_raw)

        scores = harmonic_f1(p_bank, p_user)
        best_idx = np.argsort(scores)[-top_k:][::-1]

        client_results = {}
        for rank, i in enumerate(best_idx, 1):
            slot = time_grid.iloc[i]
            dow = int(slot['dow'])
            hour = int(slot['hour'])
            dow_name = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'][dow]

            client_results.update({
                f'best_time_rank_{rank}_dow': dow,
                f'best_time_rank_{rank}_hour': hour,
                f'best_time_rank_{rank}_dow_name': dow_name,
                f'best_time_rank_{rank}_time_str': f"{dow_name} {hour:02d}:00",
                f'best_time_rank_{rank}_harmonic_f1': round(float(scores[i]), 5),
                f'best_time_rank_{rank}_p_bank': round(float(p_bank[i]), 5),
                f'best_time_rank_{rank}_p_user': round(float(p_user[i]), 5),
            })

        if 'client_id' in client_row:
            client_results['client_id'] = client_row['client_id']

        results.append(client_results)

    result_df = pd.DataFrame(results)
    if 'client_id' in df.columns:
        final_df = df[['client_id']].reset_index(drop=True).merge(result_df, on='client_id', how='left')
    else:
        final_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    return final_df