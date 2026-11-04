"""Explaining the gradient-boosting forecasts: split-gain importance, feature groups and SHAP values."""
import numpy as np
import pandas as pd

from . import features as F

FEATURE_GROUPS = {
    "Calendar": ["dow", "is_weekend", "day_of_month", "month", "week_of_year", "day_of_year", "days_to_month_end", "month_part",
                 "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "is_christmas_eve", "is_new_years_eve", "horizon_day"],
    "Holidays": ["state_holiday", "school_holiday", "days_to_holiday", "days_since_holiday", "days_to_easter", "days_since_easter",
                 "days_to_christmas", "days_since_christmas"],
    "Promotion": ["promo", "promo_dow", "promo_week", "promo_prev_week"],
    "Opening and closures": ["days_since_closure", "days_since_long_closure", "days_to_long_closure", "days_since_gap_end"],
    "Competition and Promo2": ["comp_open_months", "comp_not_yet_open", "comp_opened_last_6m", "comp_recent_near", "promo2_active",
                               "promo2_weeks", "promo2_start_month", "competition_distance_log", "competition_distance_missing", "promo2",
                               "promo_interval"],
    "Store details": ["store_type", "assortment", "has_history_gap"],
    "History: store level": ["hist_mean_28", "hist_mean_91", "hist_mean_all", "hist_customers_91", "hist_spc_91", "hist_days_91",
                             "hist_dow_mean_91", "hist_dow_mean_all"],
    "History: promotion and trend": ["hist_promo_mean_91", "hist_nopromo_mean_91", "hist_promo_lift_all", "hist_momentum"],
    "History: last year": ["ly_sales_same_dow", "ly_week_mean"],
}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


def unwrap(model):
    """(preprocessor, HistGradientBoostingRegressor) from the TransformedTargetRegressor pipeline of src/models.py."""
    pipe = model.regressor_
    return pipe.named_steps["prep"], pipe.named_steps["model"]


def original_feature(names, features):
    """Map each preprocessed column name back to the model input it came from (one-hot and missing flags included)."""
    cats = sorted(F.split_by_kind(features)["categorical"], key=len, reverse=True)
    out = []
    for n in names:
        if n.startswith("missingindicator_"):
            out.append(n[len("missingindicator_"):])
        elif n in features:
            out.append(n)
        else:
            out.append(next(c for c in cats if n.startswith(c + "_")))
    return np.array(out)


def gain_importance(model, features):
    """Total split gain per input feature, summed over all trees (the impurity importance of a boosted model).

    scikit-learn does not expose feature_importances_ for HistGradientBoosting, so the gain is read from the fitted
    trees (attribute `_predictors`, stable since scikit-learn 0.21 but private).
    """
    pre, hgb = unwrap(model)
    names = pre.get_feature_names_out()
    gains = np.zeros(len(names))
    for iteration in hgb._predictors:
        for tree in iteration:
            nodes = tree.nodes
            split = nodes["is_leaf"] == 0
            np.add.at(gains, nodes["feature_idx"][split], nodes["gain"][split])
    s = pd.Series(gains, index=original_feature(names, features)).groupby(level=0).sum()
    return (s / s.sum()).reindex(features).fillna(0.0)


def shap_values(model, X, features):
    """SHAP values (log-sales scale) for the rows of X, summed back to the input features. Returns (values, base value)."""
    import shap
    pre, hgb = unwrap(model)
    Xt = pre.transform(X[features])
    explainer = shap.TreeExplainer(hgb)
    sv = explainer.shap_values(Xt, check_additivity=False)
    orig = original_feature(pre.get_feature_names_out(), features)
    df = pd.DataFrame(sv, columns=orig).T.groupby(level=0).sum().T[features]
    return df, float(np.ravel(explainer.expected_value)[0])
