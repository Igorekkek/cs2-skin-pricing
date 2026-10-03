import numpy as np
import pandas as pd

from skinprice.research.trading import BacktestConfig, apply_curve, resale_after_ban, strategy_row


def _sales(rows):
    frame = pd.DataFrame(rows, columns=["name", "day", "float_value", "price_cents"])
    frame["event_time"] = pd.Timestamp("2026-07-01", tz="UTC") + pd.to_timedelta(frame["day"], unit="D")
    frame["curve_part"] = 0.0
    frame["base_equiv"] = frame["price_cents"].astype(float)
    return frame


def test_resale_uses_only_comparables_after_the_ban():
    sales = _sales(
        [
            ("A", 0, 0.100, 100),  # candidate
            ("A", 3, 0.100, 1),  # inside the trade ban: ignored
            ("A", 8, 0.105, 120),
            ("A", 9, 0.109, 130),
            ("A", 10, 0.095, 140),
            ("A", 11, 0.300, 999),  # float too far: ignored
            ("A", 20, 0.100, 999),  # after the resale window: ignored
            ("B", 8, 0.100, 999),  # another item: ignored
        ]
    )
    candidates = np.array([True] + [False] * 7)
    out = resale_after_ban(sales, candidates, BacktestConfig())
    assert out.loc[0, "resale_med"] == 130
    assert out.loc[0, "n_comparables"] == 3
    assert out["resale_med"].iloc[1:].isna().all()


def test_too_few_comparables_gives_no_estimate():
    sales = _sales([("A", 0, 0.1, 100), ("A", 8, 0.1, 120), ("A", 9, 0.1, 130)])
    out = resale_after_ban(sales, np.array([True, False, False]), BacktestConfig())
    assert np.isnan(out.loc[0, "resale_med"])


def test_strategy_roi_accounts_for_fee():
    trades = pd.DataFrame({"price_cents": [100.0, 100.0], "resale": [110.0, 110.0]})
    row = strategy_row(trades, trades, "resale", fee=0.02, days=7)
    assert row["roi_pct"] == round(100 * (110 * 0.98 - 100) / 100, 2)


def test_curve_interpolates_in_log_space():
    xs, betas = np.array([-2.0, 0.0]), np.array([0.5, 0.1])
    assert np.allclose(apply_curve(np.array([0.1, 0.0]), xs, betas), [0.3, 0.0])
