"""Two-layer LSTM forecasters (Keras 3) built on the lazy windows of src/sequences.py.

Keras 3 runs on TensorFlow (the default, and what Colab provides), PyTorch or JAX. Only `keras.ops` is used, so
the same code runs on any backend; set the KERAS_BACKEND environment variable before the first import to choose.
"""
import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # hide TensorFlow's start-up and retracing notices

import keras
import numpy as np
import pandas as pd
from keras import layers, ops

from . import sequences as SQ
from .logger import get_logger

logger = get_logger("rossmann.lstm")
if keras.backend.backend() == "tensorflow":
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")


def framework_info():
    """Backend, versions and whether a GPU is visible."""
    info = {"keras": keras.__version__, "backend": keras.backend.backend(), "gpu": "none"}
    try:
        if info["backend"] == "tensorflow":
            import tensorflow as tf
            info["tensorflow"] = tf.__version__
            gpus = tf.config.list_physical_devices("GPU")
            info["gpu"] = ", ".join(g.name for g in gpus) if gpus else "none"
        elif info["backend"] == "torch":
            import torch
            info["torch"] = torch.__version__
            info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception as err:          # informative only
        info["gpu"] = f"unknown ({err})"
    return info


def set_seeds(seed):
    """Python, NumPy and backend seeds in one call (results are repeatable on CPU; GPU kernels may still vary)."""
    keras.utils.set_random_seed(int(seed))


# ----------------------------------------------------------------------------- loss
@keras.saving.register_keras_serializable(package="rossmann")
def masked_mse(y_true, y_pred):
    """Mean squared error over scoreable days only. y_true[..., 0] = target, y_true[..., 1] = 1 if scoreable."""
    target, mask = y_true[..., 0], y_true[..., 1]
    err = ops.square(target - y_pred[..., 0]) * mask
    return ops.sum(err) / ops.maximum(ops.sum(mask), 1.0)


# ----------------------------------------------------------------------------- data feed
class WindowFeed(keras.utils.PyDataset):
    """Feeds batches cut lazily from the panel (no window is ever stored)."""

    def __init__(self, panel, samples, window, horizon, last_target_day, batch_size=256, shuffle=False, seed=42,
                 use_store=False, use_static=False, samples_per_epoch=None, **kwargs):
        """samples_per_epoch: if set (training only), each epoch uses a fresh random subset of that many samples."""
        super().__init__(**kwargs)
        self.panel, self.window, self.horizon, self.last_target_day = panel, window, horizon, last_target_day
        self.s = samples["series_idx"].to_numpy()
        self.o = samples["origin_idx"].to_numpy()
        self.batch_size, self.shuffle = batch_size, shuffle
        self.use_store, self.use_static = use_store, use_static
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.s))
        self.samples_per_epoch = min(samples_per_epoch, len(self.s)) if samples_per_epoch else len(self.s)
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(self.samples_per_epoch / self.batch_size))

    def inputs(self, idx):
        b = SQ.get_batch(self.panel, self.s[idx], self.o[idx], self.window, self.horizon, self.last_target_day)
        x = {"history": b["history"], "future": b["future"]}
        if self.use_store:
            x["store"] = b["series"].astype("int32")
        if self.use_static:
            x["static"] = SQ.static_inputs(self.panel, b["series"])
        mask = np.isfinite(b["target"])
        y = np.stack([np.nan_to_num(b["target"], nan=0.0), mask.astype("float32")], axis=-1)
        return x, y

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:min((i + 1) * self.batch_size, self.samples_per_epoch)]
        return self.inputs(idx)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


# ----------------------------------------------------------------------------- model
def build_lstm(window, horizon, n_hist, n_future, units=64, dropout=0.1, head="per_day", n_stores=None,
               embed_dim=8, n_static=0, dense_units=64, learning_rate=1e-3, name="lstm"):
    """Two stacked LSTM layers encode the history window; a direct head turns that summary plus the planned inputs
    of each future day into all `horizon` forecasts at once.

    head="per_day": the same small dense network is applied to every future day (summary + that day's plan);
    head="flat"   : all future plans are flattened and one dense layer outputs the whole horizon.
    """
    hist_in = keras.Input((window, n_hist), name="history")
    fut_in = keras.Input((horizon, n_future), name="future")
    inputs = [hist_in, fut_in]
    x = layers.LSTM(units, return_sequences=True, name="lstm_1")(hist_in)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units, name="lstm_2")(x)
    context = [x]
    if n_stores:
        st_in = keras.Input((), dtype="int32", name="store")
        inputs.append(st_in)
        context.append(layers.Flatten()(layers.Embedding(n_stores, embed_dim, name="store_embedding")(st_in)))
    if n_static:
        sx_in = keras.Input((n_static,), name="static")
        inputs.append(sx_in)
        context.append(sx_in)
    ctx = layers.Concatenate()(context) if len(context) > 1 else context[0]
    ctx = layers.Dense(dense_units, activation="relu", name="context")(ctx)
    ctx = layers.Dropout(dropout)(ctx)
    if head == "per_day":
        rep = layers.RepeatVector(horizon)(ctx)
        z = layers.Concatenate(axis=-1)([rep, fut_in])
        z = layers.TimeDistributed(layers.Dense(dense_units, activation="relu"), name="day_dense_1")(z)
        z = layers.TimeDistributed(layers.Dense(dense_units // 2, activation="relu"), name="day_dense_2")(z)
        out = layers.TimeDistributed(layers.Dense(1), name="forecast")(z)
    elif head == "flat":
        z = layers.Concatenate()([ctx, layers.Flatten()(fut_in)])
        z = layers.Dense(dense_units * 2, activation="relu")(z)
        out = layers.Reshape((horizon, 1), name="forecast")(layers.Dense(horizon)(z))
    else:
        raise ValueError(head)
    model = keras.Model(inputs, out, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0), loss=masked_mse)
    return model


# ----------------------------------------------------------------------------- re-using compiled models
class ModelPool:
    """Build each architecture once and re-use it for every split and seed.

    Building and fitting many Keras models in one TensorFlow process leaks memory (about 50 MB per fitted model in
    this project; keras.backend.clear_session does not release it), which would exhaust a Colab session over the 80
    fits of Section 8. Re-using one compiled model per architecture keeps memory nearly flat: before each fit, every
    model variable (weights and random-generator states) is set to the values of a freshly built model with that fit's
    seed (built once per architecture and seed, then cached as arrays) and the optimiser state is cleared, so each fit
    starts exactly as a new model would.
    """

    def __init__(self):
        self._models, self._init = {}, {}

    def get(self, key, builder, seed):
        set_seeds(seed)
        if (key, seed) not in self._init:
            fresh = builder()
            self._init[(key, seed)] = [np.array(v) for v in fresh.variables]
            if key not in self._models:
                self._models[key] = fresh
                return fresh
            del fresh
        model = self._models[key]
        for var, value in zip(model.variables, self._init[(key, seed)]):
            var.assign(value)
        for v in model.optimizer.variables:
            if getattr(v, "path", v.name).endswith("learning_rate"):
                continue
            v.assign(ops.zeros_like(v))
        return model


# ----------------------------------------------------------------------------- training
def split_early_stopping(samples, share=0.1, min_origins=2):
    """Chronological hold-out INSIDE the training period: the latest origins (by date) are used for early stopping."""
    origins = np.sort(samples["origin"].unique())
    n_es = max(min_origins, int(round(share * len(origins))))
    cut = origins[-n_es]
    return samples[samples["origin"] < cut].reset_index(drop=True), samples[samples["origin"] >= cut].reset_index(drop=True)


def fit(model, train_feed, es_feed, epochs, patience, checkpoint_path, verbose=0):
    """Adam with early stopping on the chronological hold-out; the best weights are kept and checkpointed.

    The checkpoint holds the weights only (<name>.weights.h5): writing the full model at every improvement would
    cost more time than the training steps of the small chain-level models.
    """
    checkpoint_path = str(checkpoint_path)
    if not checkpoint_path.endswith(".weights.h5"):
        checkpoint_path = checkpoint_path.rsplit(".", 1)[0] + ".weights.h5"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]
    t0 = time.time()
    hist = model.fit(train_feed, validation_data=es_feed, epochs=epochs, callbacks=callbacks, verbose=verbose)
    seconds = time.time() - t0
    h = pd.DataFrame(hist.history)
    h.index = np.arange(1, len(h) + 1)
    return h, seconds


def predict(model, feed, verbose=0):
    """Model outputs (n, horizon) for every sample of a feed, in the feed's order (feed must not be shuffled)."""
    return np.concatenate([model.predict_on_batch(feed[i][0]) for i in range(len(feed))], axis=0)[..., 0]


def recursive_forecast(model, panel, series_idx, origin_idx, window, steps):
    """Use a one-step model recursively: each forecast becomes part of the history for the next step.

    The planned inputs of every future day are known; only the sales channel is filled with the model's own forecasts.
    """
    s = np.asarray(series_idx, dtype=int)
    o = np.asarray(origin_idx, dtype=int)
    if len(np.unique(s)) != len(s):
        raise ValueError("recursive_forecast needs one origin per series (forecasts are written into a copy of the series)")
    work = panel.scaled.copy()
    out = np.zeros((len(s), steps), dtype="float32")
    for k in range(steps):
        tmp = SQ.Panel(panel.dates, panel.stores, panel.sales, panel.cov, panel.cov_names, panel.level,
                       panel.last_sales_date, panel.log_sales, panel.static, work, panel.scaling)
        b = SQ.get_batch(tmp, s, o + k, window, 1, panel.dates[-1])
        z = model.predict_on_batch({"history": b["history"], "future": b["future"]})[:, 0, 0]
        out[:, k] = z
        work[s, o + k + 1] = z
    return out
