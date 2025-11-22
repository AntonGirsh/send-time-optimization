import typer
from src.pipeline import full_pipeline

app = typer.Typer()

@app.command()
def train(data_path: str = typer.Option(None, "--data", "-d"),
          use_generated: bool = typer.Option(False, "--gen"),
          output_dir: str = typer.Option("models", "--out", "-o")):
    models = full_pipeline(df_path=data_path, use_generated=use_generated, output_dir=output_dir)
    typer.echo(f"Модели сохранены в {output_dir}")

if __name__ == "__main__":
    app()