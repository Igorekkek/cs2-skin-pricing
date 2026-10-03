import pandas as pd
import pytest

from skinprice import paths
from skinprice.data.io import load_tables
from skinprice.features.build import FeatureConfig, build_feature_table


@pytest.fixture(scope="session")
def sample_tables() -> dict[str, pd.DataFrame]:
    return load_tables(paths.SAMPLE_DIR)


@pytest.fixture(scope="session")
def sample_features(sample_tables) -> pd.DataFrame:
    return build_feature_table(sample_tables, FeatureConfig())
