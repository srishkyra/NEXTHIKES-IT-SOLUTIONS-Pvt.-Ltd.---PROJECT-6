"""Section 7 reference features, baselines and ML pipelines.
No short-lag recursion; delayed expanding history uses calendar dates.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

CAT7 = ['Store', 'DayOfWeek', 'Month', 'StateHoliday', 'StoreType', 'Assortment',
        'PromoInterval', 'PromoWeekday']
NUM7 = ['Promo', 'SchoolHoliday', 'Weekend', 'Day', 'Year', 'WeekOfYear',
        'MonthStart', 'MonthMid', 'MonthEnd', 'DoySin', 'DoyCos', 'TrendDays',
        'CompetitionDistance', 'CompetitionUnknown', 'CompetitionAgeMonths',
        'Promo2', 'Promo2AgeWeeks', 'Promo2Active', 'Promo2Unknown']
HIST7 = ['HistoryStoreMean', 'HistoryWeekdayMean', 'HistoryPromoMean']

def eligible7(d):
    return d.loc[d.Open.eq(1) & d.Sales.gt(0)].copy()

def score7(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if not (np.isfinite(y).all() and np.isfinite(pred).all() and (pred >= 0).all()):
        raise ValueError('Nonfinite values or negative predictions.')
    positive = y > 0
    return {'rmspe': float(np.sqrt(np.mean(((pred[positive]-y[positive])/y[positive])**2)))
                       if positive.any() else np.nan,
            'mae': float(np.mean(np.abs(pred-y))),
            'rmse': float(np.sqrt(np.mean((pred-y)**2))),
            'n': int(len(y)), 'n_positive': int(positive.sum())}

def apply_open7(pred, query):
    if not query.Open.isin([0, 1]).all():
        raise ValueError('Opening plans must be resolved before prediction.')
    out = np.maximum(np.asarray(pred, float), 0)
    out[query.Open.to_numpy() == 0] = 0
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite forecast.')
    return out

def baseline7(history, query):
    """Repeat the last observed calendar week across all forecast weeks.
    Exact absent dates fall back to store-weekday, store, then global mean.
    Group means use positive open-day sales only; seasonal lookups retain closures.
    """
    h = eligible7(history)
    global_mean = h.Sales.mean()
    if not np.isfinite(global_mean):
        raise ValueError('No eligible history for baselines.')
    store_mean = query.Store.map(h.groupby('Store').Sales.mean()).fillna(global_mean)
    weekday_map = h.groupby(['Store', 'DayOfWeek']).Sales.mean()
    idx = pd.MultiIndex.from_frame(query[['Store', 'DayOfWeek']])
    weekday = pd.Series(weekday_map.reindex(idx).to_numpy(), index=query.index).fillna(store_mean)
    promo_map = h.groupby(['Store', 'DayOfWeek', 'Promo']).Sales.mean()
    pidx = pd.MultiIndex.from_frame(query[['Store', 'DayOfWeek', 'Promo']])
    promo = pd.Series(promo_map.reindex(pidx).to_numpy(), index=query.index).fillna(weekday)
    origin = query.Date.min()
    source_dates = origin - pd.Timedelta(days=7) + pd.to_timedelta(
        (query.Date-origin).dt.days % 7, unit='D')
    lookup = history.set_index(['Store', 'Date']).Sales
    sidx = pd.MultiIndex.from_arrays([query.Store, source_dates])
    seasonal_raw = pd.Series(lookup.reindex(sidx).to_numpy(), index=query.index)
    seasonal = seasonal_raw.fillna(weekday)
    return {'SeasonalWeek': apply_open7(seasonal, query),
            'StoreWeekday': apply_open7(weekday, query),
            'PromoAware': apply_open7(promo, query)}, int(seasonal_raw.isna().sum())

def delayed_mean7(history, query, keys, delay=48):
    """As-of lookup of cumulative means. A row dated s becomes available on s+delay."""
    h = eligible7(history).sort_values(['Date', 'Store']).copy()
    q = query[['Date'] + keys].copy().reset_index(drop=True)
    q['_row'] = np.arange(len(q))
    # Composite categorical keys avoid ordering assumptions across stores.
    h['_key'] = h[keys].astype(str).agg('|'.join, axis=1)
    q['_key'] = q[keys].astype(str).agg('|'.join, axis=1)
    g = h.groupby('_key', sort=False)
    h['_mean'] = g.Sales.cumsum() / (g.cumcount() + 1)
    h['_available'] = h.Date + pd.Timedelta(days=delay)
    result = pd.merge_asof(q.sort_values('Date'),
                          h[['_key', '_available', '_mean']].sort_values('_available'),
                          left_on='Date', right_on='_available', by='_key',
                          direction='backward', allow_exact_matches=True)
    return result.sort_values('_row')['_mean'].to_numpy()

def features7(query, metadata, history=None, delay=48):
    q = query.drop(columns=['Sales', 'Customers', 'Id'], errors='ignore').copy()
    q = q.sort_values(['Date', 'Store']).reset_index(drop=True)
    x = q.merge(metadata, on='Store', how='left', validate='many_to_one', sort=False)
    assert len(x) == len(q)
    dt = x.Date.dt
    x['Month'] = dt.month
    x['Weekend'] = (dt.dayofweek >= 5).astype(int)
    x['Day'], x['Year'], x['WeekOfYear'] = dt.day, dt.year, dt.isocalendar().week.astype(int)
    x['MonthStart'] = (dt.day <= 10).astype(int)
    x['MonthMid'] = dt.day.between(11, 20).astype(int)
    x['MonthEnd'] = (dt.day > 20).astype(int)
    x['DoySin'] = np.sin(2*np.pi*dt.dayofyear/365.25)
    x['DoyCos'] = np.cos(2*np.pi*dt.dayofyear/365.25)
    x['TrendDays'] = (x.Date-pd.Timestamp('2013-01-01')).dt.days
    x['PromoWeekday'] = x.Promo.astype(str)+'_'+x.DayOfWeek.astype(str)
    x['CompetitionUnknown'] = x.CompetitionDistance.isna().astype(int)
    age = ((dt.year-x.CompetitionOpenSinceYear)*12 + dt.month-x.CompetitionOpenSinceMonth)
    x['CompetitionAgeMonths'] = age.clip(lower=0)  # no negative age before opening
    # ISO week-year, not Gregorian Jan 1 + arbitrary week offsets.
    valid = x.Promo2.eq(1) & x.Promo2SinceYear.notna() & x.Promo2SinceWeek.notna()
    promo_start = pd.Series(pd.NaT, index=x.index, dtype='datetime64[ns]')
    iso = (x.loc[valid, 'Promo2SinceYear'].astype(int).astype(str)+'-W'+
           x.loc[valid, 'Promo2SinceWeek'].astype(int).astype(str).str.zfill(2)+'-1')
    promo_start.loc[valid] = pd.to_datetime(iso, format='%G-W%V-%u', errors='raise')
    x['Promo2AgeWeeks'] = ((x.Date-promo_start).dt.days/7).clip(lower=0)
    x.loc[x.Promo2.eq(0), 'Promo2AgeWeeks'] = 0
    x['Promo2Unknown'] = (x.Promo2.eq(1) &
        (promo_start.isna() | x.PromoInterval.isna())).astype(int)
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'}
    current = dt.month.map(month_names)
    intervals = x.PromoInterval.fillna('').str.split(',')
    active_month = np.array([m in items for m,items in zip(current, intervals)])
    x['Promo2Active'] = (valid & (x.Date >= promo_start) & active_month).astype(int)
    x['PromoInterval'] = x.PromoInterval.fillna('NotParticipatingOrUnknown')
    if history is not None:
        for keys, name in [(['Store'], HIST7[0]), (['Store','DayOfWeek'], HIST7[1]),
                           (['Store','Promo'], HIST7[2])]:
            x[name] = delayed_mean7(history, q, keys, delay)
    for col in CAT7:
        x[col] = x[col].fillna('Unknown').astype(str)
    columns = CAT7 + NUM7 + (HIST7 if history is not None else [])
    return x[columns].replace([np.inf, -np.inf], np.nan)

def pipeline7(family, params, history=True, seed=42):
    numeric = NUM7 + (HIST7 if history else [])
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale', StandardScaler())]), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', dtype=np.float32), CAT7)
    ], sparse_threshold=1.0)
    if family == 'Ridge':
        model = Ridge(solver='lsqr', **params)
    elif family == 'RandomForest':
        model = RandomForestRegressor(random_state=seed, n_jobs=2, **params)
    elif family == 'LightGBM':
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=seed, n_jobs=2, verbosity=-1,
                              deterministic=True, force_col_wise=True, **params)
    else:
        raise ValueError(family)
    # Predictions returned on original sales scale; no post-hoc bias correction.
    return TransformedTargetRegressor(
        regressor=Pipeline([('pre', pre), ('model', model)]),
        func=np.log1p, inverse_func=np.expm1, check_inverse=False)
