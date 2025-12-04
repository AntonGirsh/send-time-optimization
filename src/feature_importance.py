# src/feature_importance.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import typer

app = typer.Typer(help="Анализ важности фичей в bank_model и user_model")


@app.command()
def show(
    run_id: str = typer.Argument(..., help="ID раннинга, например 2025-12-01_v2"),
    top_n: int = typer.Option(20, "--top", "-t", help="Сколько топ-фичей показать"),
    save_plots: bool = typer.Option(True, "--save", help="Сохранить графики в reports/"),
):
    """
    Показывает feature importance для обеих моделей из указанного раннинга
    """
    model_dir = Path("models") / run_id

    if not model_dir.exists():
        typer.echo(f"Ошибка: папка models/{run_id} не найдена!")
        raise typer.Exit(code=1)

    # Загружаем конфиг (там список фичей)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        typer.echo("Ошибка: config.json не найден. Убедись, что обучение прошло с --save-config")
        raise typer.Exit(code=1)

    with open(config_path) as f:
        config = json.load(f)

    feature_names = config["features"]

    # Загружаем важности (CatBoost сохраняет их в JSON при learning_rate или через get_feature_importance)
    bank_path = model_dir / "bank_model_feature_importance.json"
    user_path = model_dir / "user_model_feature_importance.json"

    if not bank_path.exists() or not user_path.exists():
        typer.echo("Ошибка: файлы *_feature_importance.json не найдены.")
        typer.echo("   Добавь в train.py после fit: model.save_feature_importance('bank_model_feature_importance.json')")
        raise typer.Exit(code=1)

    bank_imp = pd.read_json(bank_path).set_index("feature")["importance"]
    user_imp = pd.read_json(user_path).set_index("feature")["importance"]

    # Приводим к одному порядку фичей
    bank_imp = bank_imp.reindex(feature_names).fillna(0)
    user_imp = user_imp.reindex(feature_names).fillna(0)

    bank_top = bank_imp.sort_values(ascending=False).head(top_n)
    user_top = user_imp.sort_values(ascending=False).head(top_n)

    # Вывод в консоль
    typer.echo(f"\nBANK MODEL — топ-{top_n} фичей (влияние на отклик банка):")
    typer.echo(bank_top.round(4).to_string())

    typer.echo(f"\nUSER MODEL — топ-{top_n} фичей (влияние на отклик юзера):")
    typer.echo(user_top.round(4).to_string())

    if save_plots:
        plt.style.use("seaborn-v0_8")
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Bank model
        plt.figure(figsize=(10, max(6, top_n * 0.35)))
        sns.barplot(x=bank_top.values, y=bank_top.index, palette="Blues_d")
        plt.title(f"Feature Importance — bank_model ({run_id})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(reports_dir / f"{run_id}_bank_importance.png", dpi=150)
        plt.close()

        # User model
        plt.figure(figsize=(10, max(6, top_n * 0.35)))
        sns.barplot(x=user_top.values, y=user_top.index, palette="Reds_d")
        plt.title(f"Feature Importance — user_model ({run_id})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(reports_dir / f"{run_id}_user_importance.png", dpi=150)
        plt.close()

        typer.echo(f"\nГрафики сохранены:")
        typer.echo(f"   reports/{run_id}_bank_importance.png")
        typer.echo(f"   reports/{run_id}_user_importance.png")

    typer.echo("\nГотово!")


if __name__ == "__main__":
    app()