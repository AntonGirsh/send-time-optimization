# src/interpretation.py
import pandas as pd
import numpy as np
from scipy import stats
import typer

app = typer.Typer(help="Интерпретация: надёжные сегменты с высоким uplift")


def detect_feature_type(series: pd.Series) -> str:
    if series.nunique() <= 16 or series.dtype == 'object' or series.dtype.name == 'category':
        return 'categorical'
    return 'numerical'


def cohen_d(g1: pd.Series, g2: pd.Series) -> float:
    diff = g1.mean() - g2.mean()
    pooled_std = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
    return diff / pooled_std if pooled_std > 0 else 0.0


@app.command()
def profile_top_slots(
    pred_parquet: str = typer.Option("reports/pred.parquet", "--pred", "-p", help="Путь к предиктам"),
    data_parquet: str = typer.Option("data/synthetic.parquet", "--data", "-d", help="Путь к данным"),
    min_uplift: float = 0.07,           # +7% по умолчанию (в долях)
    target_coverage: float = 0.50,      # остановиться при 50%
    min_clients_abs: int = 1000,        # минимум клиентов в слоте
    min_clients_rel: float = 0.01,      # минимум % от всей базы
):
    pred = pd.read_parquet(pred_parquet)
    data = pd.read_parquet(data_parquet)

    # Приводим к правильным типам
    for col in ['best_dow', 'best_hour']:
        pred[col] = pd.to_numeric(pred[col], errors='coerce').fillna(-1).astype(int)
    pred['uplift_pct'] = pd.to_numeric(pred['uplift_pct'], errors='coerce').fillna(0)

    df = pred.merge(
        data.drop(columns=['accept_bank', 'accept_user'], errors='ignore'),
        on='client_id', how='left'
    )

    # Определяем фичи для анализа
    exclude = {'client_id', 'best_dow', 'best_hour', 'best_score', 'random_score', 'uplift_abs', 'uplift_pct', 'send_ts'}
    feature_types = {c: detect_feature_type(df[c]) for c in df.columns if c not in exclude}

    # Статистика по слотам
    slot_stats = (
        df.groupby(['best_dow', 'best_hour'])
        .agg(uplift_mean=('uplift_pct', 'mean'), clients=('client_id', 'nunique'))
        .reset_index()
    )

    total_clients = len(df)

    # 1. Фильтр по минимальному uplift
    candidates = slot_stats[slot_stats['uplift_mean'] > min_uplift].copy()

    # 2. Фильтр по размеру слота
    min_clients = max(min_clients_abs, min_clients_rel * total_clients)
    reliable_slots = candidates[candidates['clients'] >= min_clients].sort_values('uplift_mean', ascending=False)

    if reliable_slots.empty:
        typer.echo(f"Нет надёжных слотов: uplift > +{min_uplift:.0%} и ≥ {int(min_clients)} клиентов.")
        typer.echo("Попробуй снизить --min-uplift или --min-clients-abs")
        return

    typer.echo(f"Надёжных слотов: {len(reliable_slots)} (uplift > +{min_uplift:.0%}, ≥ {int(min_clients)} клиентов)\n")
    typer.echo(f"Профилируем до покрытия {target_coverage:.0%} клиентов...\n")

    covered = 0
    dow_names = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}

    for _, row in reliable_slots.iterrows():
        dow = int(row['best_dow'])
        hour = int(row['best_hour'])
        mask = (df['best_dow'] == dow) & (df['best_hour'] == hour)
        segment = df[mask]

        covered += len(segment)

        typer.echo(f"{dow_names.get(dow, '??')} {hour:02d}:00")
        typer.echo(f"   Клиентов: {len(segment):,} ({len(segment)/total_clients:.1%})")
        typer.echo(f"   Uplift: +{row['uplift_mean']:.1%} | Покрыто: {covered/total_clients:.1%}")

        pop = df
        for feat, ftype in feature_types.items():
            s = segment[feat].dropna()
            p = pop[feat].dropna()
            if len(s) < 30 or len(p) < 30:
                continue

            if ftype == 'numerical':
                med_s, med_p = s.median(), p.median()
                d = cohen_d(s, p)
                p_val = stats.mannwhitneyu(s, p, alternative='two-sided').pvalue
                if abs(d) >= 0.5 and p_val < 0.001:
                    dir_text = "гораздо выше" if d > 0 else "гораздо ниже"
                    typer.echo(f"   • {feat:20} → {med_s:.2f} (vs {med_p:.2f}) | {dir_text}")
                elif abs(d) >= 0.3 and p_val < 0.01:
                    dir_text = "выше" if d > 0 else "ниже"
                    typer.echo(f"   • {feat:20} → {med_s:.2f} (vs {med_p:.2f}) | {dir_text}")

            else:  # categorical
                if s.mode().empty:
                    continue
                top = s.mode().iloc[0]
                s_share = (s == top).mean()
                p_share = (p == top).mean()
                if s_share > p_share + 0.08:
                    chi2_p = stats.chi2_contingency([
                        [(s == top).sum(), (s != top).sum()],
                        [(p == top).sum(), (p != top).sum()]
                    ], correction=False)[1]
                    if chi2_p < 0.001:
                        typer.echo(f"   • {feat:20} → {top!r}: {s_share:.0%} (vs {p_share:.0%}) | сильно больше")

        typer.echo("")

        if covered >= target_coverage * total_clients:
            typer.echo(f"Достигнуто {covered/total_clients:.1%} покрытие — готово!")
            break
    else:
        typer.echo(f"Все надёжные слоты обработаны. Покрытие: {covered/total_clients:.1%}")

    typer.echo("\nИнтерпретация завершена.")


if __name__ == "__main__":
    app()