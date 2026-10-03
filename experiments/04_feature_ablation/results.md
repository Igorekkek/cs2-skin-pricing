<!-- сгенерировано run.py -->

### Результаты на тестовой выборке

| вариант | модель | медианная ошибка, % | средняя ошибка, % | ошибка p90, % | средняя ошибка, $ | ошибка ≤10%, доля % | цена в интервале, % | размер, МБ | обучение, мин |
|---|---|---|---|---|---|---|---|---|---|
| all_default | premium_catboost | 5.03 | 11.46 | 26.71 | 1.48 | 68.68 | 67.09 | 4.62 | 6.10 |
| minus_pattern_seed | premium_catboost | 5.04 | 11.43 | 26.77 | 1.48 | 68.47 | 63.70 | 5.51 | 7.20 |
| minus_float_rank | premium_catboost | 4.98 | 11.46 | 26.83 | 1.47 | 68.50 | 66.55 | 4.19 | 5.60 |
| minus_pattern_extras | premium_catboost | 5.07 | 11.50 | 26.84 | 1.46 | 68.31 | 64.73 | 5.26 | 6.90 |
| minus_stickers | premium_catboost | 5.16 | 11.94 | 28.12 | 1.94 | 67.82 | 65.00 | 5.06 | 6.50 |
| minus_keychains | premium_catboost | 5.13 | 11.56 | 27.04 | 1.48 | 68.10 | 66.17 | 5.05 | 6.60 |
| minus_market_history | premium_catboost | 4.96 | 11.09 | 26.00 | 1.40 | 68.82 | 68.24 | 7.48 | 10.30 |
| minus_reference | premium_catboost | 5.73 | 13.54 | 31.05 | 1.96 | 64.19 | 67.49 | 3.15 | 3.90 |
| plus_lot_behavior | premium_catboost | 5.09 | 11.52 | 26.72 | 1.50 | 68.41 | 65.05 | 5.14 | 7.70 |
| plus_calendar | premium_catboost | 5.08 | 11.55 | 27.05 | 1.44 | 68.34 | 65.28 | 5.81 | 7.50 |
| all_default_seed7 | premium_catboost | 4.93 | 11.37 | 26.75 | 1.40 | 68.75 | 68.27 | 4.49 | 4.50 |
| all_default_seed123 | premium_catboost | 5.13 | 11.44 | 26.58 | 1.54 | 68.08 | 65.46 | 6.20 | 6.70 |
| full_premium_catboost | premium_catboost | 4.96 | 11.39 | 26.75 | 1.38 | 68.70 | 65.86 | 6.23 | 21.80 |
| full_premium_catboost_minus_market_history | premium_catboost | 4.90 | 11.18 | 26.45 | 1.33 | 68.73 | 68.40 | 7.68 | 21.00 |
| full_target_encoded_hgb | target_encoded_hgb | 4.65 | 11.02 | 26.12 | 1.37 | 69.42 | 71.08 | 7.77 | 5.10 |
| full_target_encoded_hgb_minus_market_history | target_encoded_hgb | 4.88 | 11.31 | 27.01 | 1.44 | 68.28 | 68.79 | 5.58 | 3.90 |

#### Только продажи, у которых есть новые поля

| вариант | подвыборка | продаж | медианная ошибка, % | средняя ошибка, % | средняя ошибка, $ |
|---|---|---|---|---|---|
| all_default | есть ранг float | 2044 | 17.16 | 28.97 | 15.43 |
| all_default | есть данные blue gem / fade | 2527 | 7.88 | 20.30 | 9.90 |
| minus_float_rank | есть ранг float | 2044 | 17.50 | 29.66 | 13.85 |
| minus_float_rank | есть данные blue gem / fade | 2527 | 7.56 | 20.87 | 9.19 |
| minus_pattern_extras | есть ранг float | 2044 | 16.98 | 28.19 | 14.34 |
| minus_pattern_extras | есть данные blue gem / fade | 2527 | 8.40 | 21.94 | 10.17 |
