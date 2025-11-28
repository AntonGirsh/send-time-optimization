import typer
from pathlib import Path
from src.data_generation import generate_and_save
from src.pipeline import train_pipeline, predict_pipeline

app = typer.Typer()

@app.command()
def data(n_samples: int = typer.Option(5000, "--n"),\
          output: Path = typer.Option("data/synthetic.parquet", "--output")):
    generate_and_save(n_samples, output)
    typer.echo(f"Сгенерировано {n_samples} строк → {output}")

@app.command()
def train(data_path: Path = typer.Argument(..., help="Путь к датасету для обучения"),\
           run_id: str = typer.Option(..., "--run-id")):
    train_pipeline(data_path, run_id)
    typer.echo(f"Обучение завершено → models/{run_id}/")

@app.command()
def predict(model_run: str = typer.Argument(..., help="Run ID обученной модели"), \
            input_data: Path = typer.Argument(..., help="Путь к датасету для инференса"),\
            output: Path = typer.Option(None, "--output")):
    if output is None:
        output = input_data.with_name(f"{input_data.stem}_predictions_{model_run}.parquet")
    predict_pipeline(model_run, input_data, output)
    typer.echo(f"Предикты сохранены → {output}")

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        typer.echo(f"Ошибка: {e}")
        raise typer.Exit(code=1)