# Forecasting Six Weeks of Sales for 1,115 Rossmann Stores

*An end-to-end machine learning project: exploration, validation, gradient boosting, a two-layer LSTM, uncertainty and a web app.*

## 1. The problem
Rossmann's finance team wants each store's daily sales six weeks ahead, and store managers want the same forecast for their
own store, with customers, in a form they can adjust. Until now, managers forecast from experience. The data: 1,017,209
store-days (January 2013 to July 2015) for 1,115 stores, store details (type, assortment, competition, a second
promotion scheme), and a 48-day test window (1 August to 17 September 2015) with the planned promotions, holidays and
opening days but no sales.

## 2. What the data says (exploration)
- **The calendar drives sales.** December runs at about 1.22 times a normal day; the three days before Easter sell about
  40% more; stores almost all close on state holidays. (Figure 5.2, Figure 5.3)
- **Promotions work everywhere, unevenly.** +39% sales, +20% customers and +16% spend per customer on ordinary weekdays;
  Monday promotions +57%, Friday +22%; type b and assortment b stores respond least. (Figure 5.5)
- **A new competitor nearby costs about a tenth of sales**: -7.1% on average over three months, -14% within 1 km.
  (Figure 5.11)
- **One promotion calendar for the whole chain**, Monday to Friday in alternating weeks, so sales repeat every two weeks.
- **The test window is a summer window**: 44% of its days are school holidays, against 18% across the training data.

## 3. Testing forecasts honestly
Every forecast is made the way the real one will be: from a single cut-off, using only history up to that day plus the
planned calendar of the forecast days. Three rolling 42-day windows (February to June 2015) are used to choose models;
a 48-day window one year before the test window (the *seasonal check*) is a summer stress test that is never used for
tuning; and the last 42 days (20 June to 31 July 2015) are a **final holdout**, locked in code and scored exactly once
after every choice was frozen. Leakage is a failed test, not a judgement call: a feature-availability policy blocks
customers and every sales value after the cut-off, and perturbation tests prove that scrambling future sales changes no
input. The primary metric is RMSPE (root mean squared percentage error) on open days with sales; models learn
log(1 + sales), which behaves like a percentage error.

## 4. Features and machine learning
59 features in three classes: the calendar and planned inputs (weekday, holidays and the distance to them, promotions
this and last week, closures and reopenings), store details (type, assortment, competition age and distance, Promo2), and
the store's history summarised up to the cut-off (recent and long-run averages, weekday profile, promotion lift,
momentum). Gradient boosting (scikit-learn, 300 trees of up to 127 leaves) beat Random Forest and Ridge and the best
simple rule (each store's average for that weekday and promotion status) in every window. Part 2 re-checked one feature
decision of Part 1 at the final training size before the holdout: removing the two same-year-earlier features changed the
rolling error by -0.68%, so they were dropped.

## 5. Deep learning: a two-layer LSTM

### 5.1 The time series
Averaged over open stores, daily sales form one chain-level series of 900 days. The Augmented Dickey-Fuller test rejects
a unit root and the KPSS test accepts stationarity around a trend: the series is **trend-stationary**, so differencing is
not needed for stationarity. A first difference raises the variance (1.15 times the original) and so does a
weekly difference (1.68); a two-week difference lowers it (0.86). The autocorrelation peaks at lags 14, 28, 42 and 56,
and the two-week cycle is stronger than the weekly one (cycle strength 0.60 for 14 days against 0.36 for 7):
the alternating-week promotion calendar sets the rhythm. (Figure 8.3)

### 5.2 From a series to supervised samples
Each sample is a forecast origin: the last 56 days (four two-week cycles) of scaled sales and planned inputs, the planned
inputs of the next 48 days, and the 48 sales values to predict. Sales are scaled to (-1, 1) with the range of the
training days only (store level: log sales minus the store's own mean, so one network serves small and large stores).
Origins are taken every 7 days; windows are cut on demand from one compact panel (about 80 MB instead of 0.9 GB of
materialised windows). Five tests (explicit slices, perturbation, scaler fitted on training days only, split structure,
scaling round trip) all pass.

### 5.3 The brief's model: predict the next sale, then forecast six weeks
A two-layer LSTM (64 units each) that predicts the next day's chain-level sales from the previous 56 days reaches a
one-step RMSPE of 0.083 on the rolling windows, using the actual history up to the day before each forecast. A
six-week forecast does not have that history: fed its own forecasts **recursively** over the whole window, the same model
scores 0.084 on the rolling windows and 0.071 on the summer seasonal check. The **direct** model, which outputs all 48 days at
once from the cut-off and reads the planned calendar of every future day, scores 0.083 and 0.068. The two framings are within
the spread between random seeds of each other, so the choice rests on practicality: one pass instead of 48, no feedback of
errors, and a direct use of the known promotion calendar. Window length (28 / 56 / 84 days: 0.081 / 0.083 /
0.083) barely matters. Two-week differencing hurts badly (0.212 against 0.083 on levels): holiday spikes dominate the
differenced series and every forecast error is carried forward through the inversion, which confirms 8.1's conclusion that the
series is trend-stationary and needs no differencing. A flat output head with three times the parameters overfits the
roughly 90 training origins (0.265). At chain level, gradient boosting's store forecasts averaged to the same series score
0.063, better than every LSTM variant. (Figure 8.12)

### 5.4 One LSTM for all stores
The production-scale model is a single LSTM shared by all 1,115 stores: the 56-day history of a store, its plan for the
next 48 days, a learned 8-number store embedding and its type, assortment and usual level go in; a small network applied
to each future day turns the LSTM's summary and that day's plan into the forecast. It is trained on about 100,000 to
114,000 store-origins per split, with early stopping on the latest training origins. On exactly the rows gradient
boosting is scored on, its RMSPE is 0.179, 0.148, 0.125, 0.127 (rolling 1 to 3, seasonal check) against
0.172, 0.111, 0.102, 0.105 for gradient boosting: overall 14.6% higher. Its relative standing is best in
week 1 of the horizon. (Figure 8.13)

### 5.5 A fair verdict
A decision rule was written down before the comparison: a challenger replaces gradient boosting only if it wins every
window, its store-level bootstrap interval on the pooled rolling windows lies below zero, and it gains at least 3% (an
LSTM adds TensorFlow and a second model to the app). Result: the LSTM is 14.6% worse and the 50/50 ensemble
2.7% worse than gradient boosting on the pooled rolling windows; **Gradient boosting** goes forward.
The LSTM learns the calendar and the two-week rhythm from raw sequences without hand-made features, which makes it a useful check on the feature engineering, but on this data the engineered history features carry more information.
(Figure 8.7)

## 6. Results and uncertainty
On the final holdout (20 June to 31 July 2015, scored once), the production model's RMSPE is **0.1068**, against
0.1468 for the best simple rule; the target (at most 0.13 and below the baseline) is **met**. Customers are
forecast with RMSPE 0.0877. Every forecast has a 90% range from split-conformal
calibration on the rolling windows (conformal by forecast week and store size); on the holdout it contained the actual sales
88.4% of the time. The model relies most on the store's own history and the calendar; promotions come next
(Figure 9.3). (Figure 10.1, Figure 9.4)

## 7. The app
A Streamlit app lets a manager pick a store, download that store's plan for the 48 days, change promotions or opening
days, upload it and get daily sales and customer forecasts with 90% ranges, a chart and a CSV download. It builds its
inputs with the same code as the training pipeline; for all 856 test stores its forecasts equal the submission file.

## 8. Recommendations and limitations
- Use the forecasts for six-week planning, with the 90% range as the planning buffer; re-run when plans change.
- Review promotion spend first where response is weakest (type b, assortment b) and favour early-week promotion days.
- Treat state holidays, the 33 Sunday-trading stores and the three competitors opening inside the test window with extra care.
- Limits: promotion effects are associations (one shared calendar); only two and a half years of history and one full
  earlier summer; the 2014 gap of 180 stores; the LSTM was trained on a CPU budget.
