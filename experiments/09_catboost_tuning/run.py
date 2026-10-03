"""Why does CatBoost trail the sklearn booster and the MLP in experiment 02?

Candidates, one change at a time against the experiment-02 configuration:
  * has_time       - target statistics of categories computed in chronological row
                     order instead of random permutations (the data is time-ordered);
  * separate_mae   - a dedicated MAE model for the median instead of the shared
                     MultiQuantile model (from experiment 03);
  * depth8         - deeper trees (rejected in the original research; re-checked on the
                     new features).

Usage: python experiments/09_catboost_tuning/run.py [--force]
"""

from skinprice.research import Experiment

exp = Experiment("09_catboost_tuning")
features = exp.features()
exp.reuse("baseline", "02_model_comparison", "premium_catboost")
exp.variant("has_time", "premium_catboost", features, params={"has_time": True})
exp.reuse("separate_mae", "03_model_compression", "separate_ctr1_no_seed_key")
exp.variant("depth8", "premium_catboost", features, params={"depth": 8})
exp.report()
