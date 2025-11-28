# src/visualization.py
import pandas as pd
import plotly.express as px
import typer

app = typer.Typer(help="Визуализация результатов Send Time Optimization")

@app.command()
def uplift_heatmap(
    predictions_parquet: str = typer.Argument(..., help="Путь к файлу с предиктами (из predict)"),
    output_html: str = "reports/uplift_heatmap.html"
):
    df = pd.read_parquet(predictions_parquet)
    
    heatmap_data = (
        df.groupby(['best_dow', 'best_hour'])['uplift_pct']
        .mean()
        .mul(100)
        .reset_index()
    )

    fig = px.density_heatmap(
        heatmap_data,
        x='best_hour',
        y='best_dow',
        z='uplift_pct',
        color_continuous_scale='RdYlGn',
        nbinsx=24,
        nbinsy=7,
        text_auto='.1f',
        title='Прирост конверсии vs random отправка<br><sup>по лучшему времени для каждого клиента</sup>',
        labels={'uplift_pct': 'Uplift, %', 'best_hour': 'Час', 'best_dow': 'День недели'},
        height=600
    )
    
    fig.update_layout(
        yaxis=dict(
            tickvals=list(range(7)),
            ticktext=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        ),
        coloraxis_colorbar=dict(title="Uplift, %")
    )
    
    fig.write_html(output_html)
    print(f"Интерактивный график сохранён → {output_html}")
    print(f"Открывай в браузере и кидай кому угодно")

if __name__ == "__main__":
    app()