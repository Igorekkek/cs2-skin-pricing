"""Retrain on train+validation after choosing the budget on validation?

The validation split holds the four days right before the test period - and, because
the collector ramped up in late July, 82k of the most relevant sales. Leaving them out
of the final fit wastes the freshest data. Variant: choose trees/epochs with early
stopping on validation as usual, then retrain on train+validation with that budget
(calibration offsets are kept from the validation-scored model).

Usage: python experiments/08_refit_on_validation/run.py [--force]
"""

from skinprice.research import Experiment

exp = Experiment("08_refit_on_validation")
features = exp.features()
for name in ("premium_catboost", "quantile_mlp", "target_encoded_hgb"):
    exp.reuse(name, "02_model_comparison", name)
    exp.variant(f"{name}_refit", name, features, params={"refit_on_validation": True})
exp.report()
