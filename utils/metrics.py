import numpy as np

# Small epsilon guarding divisions; metrics whose denominator is degenerate
# (e.g. constant true series for RSE) return NaN instead of an arbitrary
# huge number, so aggregation code can detect and report exclusions.
EPS = 1e-8


def RSE(pred, true):
    denom = np.sqrt(np.sum((true - true.mean()) ** 2))
    if denom < EPS:
        return np.nan
    return np.sqrt(np.sum((true - pred) ** 2)) / denom


def CORR(pred, true):
    """Pearson correlation across test windows, averaged over channels.

    The Time-Series-Library original used sqrt(sum(a^2*b^2)) as denominator
    (not a correlation; values could exceed 1, hence its mysterious 0.01
    factor) — replaced with the actual Pearson formula, range [-1, 1].
    """
    a = true - true.mean(0)
    b = pred - pred.mean(0)
    u = (a * b).sum(0)
    d = np.sqrt((a ** 2).sum(0)) * np.sqrt((b ** 2).sum(0))
    d += 1e-12
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def ME(pred, true):
    """Mean error (bias): positive = overprediction, negative = underprediction."""
    return np.mean(pred - true)


def SMAPE(pred, true):
    """Symmetric MAPE in percent (0..200)."""
    denom = np.abs(pred) + np.abs(true)
    denom = np.where(denom < EPS, EPS, denom)
    return 200.0 * np.mean(np.abs(pred - true) / denom)


def WAPE(pred, true):
    """Weighted absolute percentage error: sum|err| / sum|true|."""
    denom = np.sum(np.abs(true))
    if denom < EPS:
        return np.nan
    return np.sum(np.abs(pred - true)) / denom


def MASE(pred, true):
    """MAE scaled by the naive one-step forecast on the evaluation series.

    The consecutive one-step test windows reconstruct the actual series, so
    the denominator is mean|t_i - t_{i-1}| over that series (naive persistence
    on the test segment, not the train split — documented deviation from the
    textbook in-sample definition, chosen because per-station train series are
    not available at metric time). MASE < 1 = better than persistence.
    """
    series = np.asarray(true).reshape(-1)
    if series.size < 2:
        return np.nan
    naive_mae = np.mean(np.abs(np.diff(series)))
    if naive_mae < EPS:
        return np.nan
    return MAE(pred, true) / naive_mae


def peak_mask(true, q=0.9):
    """Boolean mask of 'peak' points: true value >= q-quantile of the block."""
    return np.asarray(true) >= np.quantile(np.asarray(true), q)


def PEAK_MAE(pred, true, q=0.9):
    mask = peak_mask(true, q)
    if not mask.any():
        return np.nan
    return np.mean(np.abs(np.asarray(pred)[mask] - np.asarray(true)[mask]))


def UNDER_RATE(pred, true):
    """Share of points where the model underpredicts (pred < true)."""
    return float(np.mean(np.asarray(pred) < np.asarray(true)))


def PEAK_UNDER_RATE(pred, true, q=0.9):
    mask = peak_mask(true, q)
    if not mask.any():
        return np.nan
    return float(np.mean(np.asarray(pred)[mask] < np.asarray(true)[mask]))


def metric(pred, true):
    """Legacy 7-tuple kept for backward compatibility (metrics.npy layout).

    Note: corr is the plain Pearson-style correlation; the historic
    unexplained 0.01 scaling factor was removed.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def metric_all(pred, true):
    """Full named metric set, saved as metrics_ext.json next to metrics.npy."""
    return {
        'mae': float(MAE(pred, true)),
        'mse': float(MSE(pred, true)),
        'rmse': float(RMSE(pred, true)),
        'mape': float(MAPE(pred, true)),
        'mspe': float(MSPE(pred, true)),
        'rse': float(RSE(pred, true)),
        'corr': float(np.mean(CORR(pred, true))),
        'me': float(ME(pred, true)),
        'smape': float(SMAPE(pred, true)),
        'wape': float(WAPE(pred, true)),
        'mase': float(MASE(pred, true)),
        'peak_mae': float(PEAK_MAE(pred, true)),
        'under_rate': float(UNDER_RATE(pred, true)),
        'peak_under_rate': float(PEAK_UNDER_RATE(pred, true)),
        'n_points': int(np.asarray(true).size),
    }
