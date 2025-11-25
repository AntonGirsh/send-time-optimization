from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

def train_bank_model(X_train, y_train, X_val, y_val, cat_features):
    model = CatBoostClassifier(
        iterations=2000,
        depth=10,
        learning_rate=0.03,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
        cat_features=cat_features
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model

def train_user_model(X_train, y_train, X_val, y_val, cat_features):
    model = CatBoostClassifier(
        iterations=1500,
        depth=8,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
        cat_features=cat_features
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model

def calibrate_model(model, X_calib, y_calib):
    raw_probs = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_calib)
    return calibrator