# Сэмпл данных

Все завершённые продажи на CSFloat для детерминированно выбранных 5% оружейных скинов,
вместе с их наклейками и брелоками: 37 тысяч продаж с 2022 года по август 2026. Скин
попадает в сэмпл, если `crc32(name) % 1000 < 50`. Берутся скины целиком, а не случайные
строки, поэтому у каждого скина сохраняется вся история продаж, и признаки истории ведут
себя так же, как на полных данных. Пересобрать сэмпл: `skinprice data sample --per-mille 50`.

| файл | одна строка — это | основные столбцы |
|---|---|---|
| `sales.parquet` | продажа | `listing_id`, `name` (рыночное название предмета), `price_cents`, `reference_*` (оценка CSFloat и время её обновления), `float_value`, `paint_seed` (паттерн), `paint_index` (раскраска), `created_at` (выставлен), `sold_at` (продан), ранг float, поля blue gem / fade |
| `stickers.parquet` | наклейка на предмете | `listing_id`, `position`, `slot`, `sticker_id`, `name`, `wear` (потёртость; пусто, если наклейка не поцарапана), `reference_price_cents`, `reference_quantity`, `reference_updated_at` |
| `keychains.parquet` | брелок на предмете | `listing_id`, `position`, `keychain_id`, `name`, `pattern`, `reference_price_cents` |

Цены указаны в центах США. В CS2 наклейки можно размещать свободно, поэтому несколько
наклеек могут занимать один `slot`. Строки уникальны по паре (`listing_id`, `position`).
