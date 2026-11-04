"""MLflow serving wrapper for the production models.

MLflow's input schema has no 8- or 16-bit integer type, so it converts such columns to 32-bit integers before
predicting. In the preprocessing pipeline that moves the numeric block from float32 to float64 arithmetic; the tiny
rounding differences flip a few tree splits and can move a store-day forecast by a few hundred sales units. This
wrapper restores the training column types before predicting, so a model served through MLflow gives exactly the
same forecasts as the model in the timestamped bundle (checked in Section 10.3).
"""
import joblib
import mlflow.pyfunc
import pandas as pd


class TypedModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        saved = joblib.load(context.artifacts["model"])
        self.model, self.dtypes = saved["model"], saved["dtypes"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[list(self.dtypes)].astype(self.dtypes)
        return self.model.predict(X)


def save_for_mlflow(model, dtypes, path):
    """Write the fitted model and its training column types to one file (the MLflow artifact)."""
    joblib.dump({"model": model, "dtypes": {c: str(t) for c, t in dtypes.items()}}, path, compress=3)
    return path
