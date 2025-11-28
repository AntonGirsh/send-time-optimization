import typer
from pathlib import Path
from src.data_generation import generate_and_save
from src.pipeline import train_pipeline, predict_pipeline
from src.prediction import predict_with_uplift
from src.visualization import app as viz_app

app = typer.Typer()

@app.command()
def data(n_samples: int = typer.Option(5000, "--n"), output: Path = typer.Option("data/synthetic.parquet", "--output")):
    generate_and_save(n_samples, output)
    typer.echo(f"Сгенерировано {n_samples} строк → {output}")

@app.command()
def train(data_path: Path = typer.Argument(..., help="Путь к датасету для обучения"), run_id: str = typer.Option(..., "--run-id")):
    train_pipeline(data_path, run_id)
    typer.echo(f"Обучение завершено → models/{run_id}/")

@app.command()
def predict(
    model_run: str,
    input_data: Path,
    output: Path = Path("reports/last_predictions.parquet")
):
    df = pd.read_parquet(input_data)
    artifacts = joblib.load(f"models/{model_run}/artifacts.joblib")
    
    result_df = predict_with_uplift(artifacts, df)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output, index=False)
    typer.echo(f"Предикты + uplift сохранены → {output}")

app.add_typer(viz_app, name="viz", help="Визуализация результатов")

if __name__ == "__main__":
    app()