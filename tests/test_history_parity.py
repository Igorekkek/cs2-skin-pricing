"""Training features and inference features must be the same numbers.

History features of a training row are computed from strictly earlier sales.
At inference they come from a MarketSnapshot. If the snapshot is built from
everything before time T, a sale at time T must get identical features either
way; any difference would be train/serve skew.
"""

import numpy as np
import pandas as pd
import pytest

from skinprice.features.build import usable_rows
from skinprice.features.history import HISTORY_COLUMNS, MarketSnapshot, history_features
from skinprice.features.static import static_features


@pytest.fixture(scope="module")
def static_frame(sample_tables):
    frame = static_features(sample_tables)
    return frame[usable_rows(frame)].reset_index(drop=True)


@pytest.mark.parametrize("quantile", [0.3, 0.6, 0.85, 0.97])
def test_snapshot_reproduces_training_features(static_frame, quantile):
    frame = static_frame
    everything = pd.Series(True, index=frame.index)
    training = history_features(frame, everything)

    times = frame["event_time"].sort_values()
    cut = times.iloc[int(len(times) * quantile)]
    at_cut = frame.index[frame["event_time"] == cut]

    snapshot = MarketSnapshot.build(frame, frame["event_time"] < cut, as_of=cut)
    served = snapshot.features_for(frame.loc[at_cut])

    expected = training.loc[at_cut, HISTORY_COLUMNS].to_numpy(dtype=float)
    actual = served[HISTORY_COLUMNS].to_numpy(dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_history_ignores_same_timestamp_sales(static_frame):
    """Two sales at the same instant must not see each other's price."""
    frame = static_frame.head(200).copy()
    frame["event_time"] = frame["event_time"].iloc[0]
    hist = history_features(frame, pd.Series(True, index=frame.index))
    assert (hist["log_wt_prior_sales"] == 0).all()
    assert hist["seed_past_premium"].isna().all()
