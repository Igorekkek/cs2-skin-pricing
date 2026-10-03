"""Russian display names for result tables (results.md files and notebooks).

Code, results.json and parquet files keep the short English identifiers; only what a
reader sees is translated. Unknown names are shown unchanged.
"""

from __future__ import annotations

import re

import pandas as pd

COLUMNS = {
    # identifiers
    "variant": "вариант",
    "model": "модель",
    "subset": "подвыборка",
    "segment": "сегмент",
    "set": "набор наклеек",
    # accuracy
    "rows": "продаж",
    "median_ape": "медианная ошибка, %",
    "mean_ape": "средняя ошибка, %",
    "p90_ape": "ошибка p90, %",
    "mae_usd": "средняя ошибка, $",
    "within_10pct": "ошибка ≤10%, доля %",
    "interval_coverage": "цена в интервале, %",
    "median_ratio": "медиана прогноз/факт",
    "mean_ratio": "среднее прогноз/факт",
    "size_mb": "размер, МБ",
    "fit_minutes": "обучение, мин",
    # segments
    "price_band": "цена продажи",
    "sticker_value": "стоимость наклеек",
    "has_charm": "брелок",
    "pattern_info": "данные о паттерне",
    "share_of_rows": "доля продаж, %",
    "share_of_error": "доля ошибки в $, %",
    # experiment 05
    "sticker price lag": "задержка цены наклеек",
    "share of train crafts %": "доля крафтов в обучении, %",
    "share of test crafts %": "доля крафтов в тесте, %",
    # experiment 06
    "floor_mdape_same_day_comparables": "ошибка «соседей по дню», %",
    "model_mdape": "ошибка модели, %",
    # trading (experiment 07)
    "lots": "лотов",
    "coverage_pct": "есть оценка перепродажи, %",
    "scope": "какие лоты",
    "signal": "сигнал на покупку",
    "trades": "сделок",
    "capital_usd": "вложено, $",
    "profit_usd": "прибыль, $",
    "roi_pct": "доходность, %",
    "roi_ci90": "доходность, 90%-интервал",
    "random_roi_pct": "доходность случайных покупок, %",
    "edge_pp": "преимущество над случайными, п.п.",
    "profitable_pct": "прибыльных сделок, %",
    "avg_ticket_usd": "средняя покупка, $",
    "profit_per_month_usd": "прибыль в месяц, $",
    "time_on_market": "сколько лот висел",
    "share_pct": "доля лотов, %",
    "median_price_to_model": "медиана цена/прогноз",
    "float_tolerance": "допуск по float",
}

VALUES = {
    # segments
    "none": "нет",
    "all": "все",
    "other": "остальные",
    "charm": "с брелоком",
    "no charm": "без брелока",
    "clean": "обычный предмет",
    "craft": "крафт",
    "blue gem / fade": "blue gem / fade",
    "seen in train": "встречался в обучении",
    "novel": "новый",
    # experiment 04 / 05 subsets
    "has float rank": "есть ранг float",
    "blue gem / fade data": "есть данные blue gem / fade",
    "all test": "весь тест",
    "crafts": "все крафты",
    "fresh crafts (all sticker prices <=3d)": "крафты со свежими ценами наклеек (≤3 дн.)",
    "crafts with a price 3-30d late": "крафты с ценой наклейки на 3–30 дн. позже",
    "stale crafts (some price >30d late)": "крафты с ценой наклейки на >30 дн. позже",
    "<=3 days": "≤3 дней",
    "3-30 days": "3–30 дней",
    ">30 days or unknown": ">30 дней или неизвестно",
    # model column of ensembles
    "geometric mean of 3": "ансамбль из 3 моделей",
    "geometric mean of 2": "ансамбль из 2 моделей",
    # trading
    "all lots": "все лоты",
    "on sale >10 min": "висели >10 мин",
    "<1 min": "<1 мин",
    "1-10 min": "1–10 мин",
    "10-60 min": "10–60 мин",
    "1-6 h": "1–6 ч",
    "6-24 h": "6–24 ч",
    "1-7 d": "1–7 дн.",
    ">7 d": ">7 дн.",
}

_SIGNAL = re.compile(r"^(model|CSFloat reference) @(\d+%)$")
_SIGNAL_NAMES = {"model": "модель", "CSFloat reference": "оценка CSFloat"}
_SUFFIX = re.compile(r"^(.+) \((.+)\)$")


def value(text: str) -> str:
    """Russian name of a cell value, e.g. ``"model @15%"`` -> ``"модель, скидка ≥15%"``."""
    if text in VALUES:
        return VALUES[text]
    match = _SIGNAL.match(text)
    if match:
        return f"{_SIGNAL_NAMES[match[1]]}, скидка ≥{match[2]}"
    return text


def column(name: str) -> str:
    """Russian name of a column, also for pivoted names such as ``"rows (novel)"``."""
    if name in COLUMNS:
        return COLUMNS[name]
    match = _SUFFIX.match(name)
    if match and match[1] in COLUMNS:
        return f"{COLUMNS[match[1]]} ({value(match[2])})"
    return name


def russian(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of a result table with Russian column names and text values, for display."""
    out = table.copy()
    for name in out.columns:
        if not pd.api.types.is_numeric_dtype(out[name]) and not pd.api.types.is_datetime64_any_dtype(out[name]):
            out[name] = out[name].astype(object).map(lambda v: value(v) if isinstance(v, str) else v)
    if not isinstance(out.index, pd.RangeIndex):
        out.index = out.index.map(lambda v: value(v) if isinstance(v, str) else v)
        out.index.name = column(out.index.name) if out.index.name else None
    return out.rename(columns=lambda c: column(c) if isinstance(c, str) else c)
