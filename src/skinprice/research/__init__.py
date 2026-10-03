"""Tools for running experiments: variants, cached results, result tables.

An experiment lives in ``experiments/<slug>/`` with a ``run.py`` that defines
variants and a ``README.md`` with the hypothesis and conclusion. Each variant
is a model trained on a feature table; its metrics are appended to
``results.json`` next to the script (small, committed to git), while the
trained model and its test predictions go to ``<workspace>/runs/<slug>/``.
Re-running skips variants that already have results unless ``force=True``.

Minimal ``run.py``::

    from skinprice.research import Experiment

    exp = Experiment("05_my_hypothesis")
    features = exp.features()
    exp.variant("baseline", "premium_catboost", features)
    exp.variant("no_market_history", "premium_catboost", features,
                spec=FeatureSpec(groups=...))
    exp.report()
"""

from skinprice.research.experiment import Experiment

__all__ = ["Experiment"]
