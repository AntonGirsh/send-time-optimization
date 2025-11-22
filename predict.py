import typer
import pandas as pd
import joblib
from omegaconf import OmegaConf
from src.prediction import predict_best_time_for_dataset

app = typer.Typer()

@app.command()
def predict(data_path: str,
            models_path: str = typer.Option("models/models.pkl", "--models", "-m"),
            output_path: str = typer.Option("predictions/result.parquet", "--out", "-o"),
            top_k: int = typer.Option(3, "--top")):
    cfg = OmegaConf.load("config/base.yaml")
    models = joblib.load(models_path)
    
    df = pd.read_parquet(data_path)
    
    result_df = predict_best_time_for_dataset(
        df=df.copy(),
        time_grid=models['time_grid'],
        bank_model=models['bank_model'],
        user_model=models['user_model'],
        bank_calibrator=models['bank_calibrator'],
        user_calibrator=models['user_calibrator'],
        cfg=cfg,
        top_k=top_k
    )
    
    result_df.to_parquet(output_path, index=False)
    typer.echo(f"Готово! {len(result_df)} клиентов в {output_path}")

if __name__ == "__main__":
    app()