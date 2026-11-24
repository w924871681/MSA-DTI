import hashlib,io,json
import numpy as np
import pandas as pd
from pathlib import Path
from .features import causal_features
L=96
CPU_COLUMN=3
NON_CPU_COLUMNS=tuple(i for i in range(10) if i!=CPU_COLUMN)
RAW_COLUMNS=['CPU cores', 'CPU capacity provisioned [MHZ]', 'CPU usage [MHZ]', 'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]', 'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]', 'Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value):
    """Dtype+shape+canonical C bytes, independent of NPZ compression metadata."""
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def read_member(z, machine):
    b = z.read(machine["member"])
    if hashlib.sha256(b).hexdigest() != machine["member_sha256"]:
        raise RuntimeError("Raw VM member SHA differs from identity manifest")
    frame = pd.read_csv(io.BytesIO(b), sep=";")
    frame.columns = frame.columns.str.strip()
    if frame.columns.tolist() != ["Timestamp [ms]", *RAW_COLUMNS]:
        raise RuntimeError("Raw VM schema differs from declared 11 columns")
    a = frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    t = a[:, 0]
    if not np.all(np.isfinite(t) & (t == np.floor(t))) or np.any(np.diff(t) <= 0):
        raise RuntimeError("Selected VM violates strict native timestamp contract")
    return t.astype(np.int64), a[:, 1:]


def valid_cpu(raw):
    cpu = np.asarray(raw)[:, CPU_COLUMN]
    return np.isfinite(cpu) & (cpu >= 0)


def complete_origins(raw, horizon):
    valid = valid_cpu(raw)
    origins = np.arange(L-1, len(raw)-int(horizon), dtype=np.int64)
    keep = np.ones(len(origins), dtype=bool)
    for step in range(1, int(horizon)+1):
        keep &= valid[origins+step]
    return origins[keep]


def fit_source_normalization(source_series):
    """Fit each of nine non-CPU log scales on source20 first24h only."""
    scales = [None]*10
    counts = [0]*10
    for col in NON_CPU_COLUMNS:
        parts = [raw[np.isfinite(raw[:, col]) & (raw[:, col] >= 0), col] for _, raw in source_series]
        v = np.concatenate(parts)
        if len(v) == 0:
            raise RuntimeError("A raw non-CPU channel has no valid source observation")
        scales[col] = max(float(np.quantile(np.log1p(v), .99, method="linear")), 1e-6)
        counts[col] = int(len(v))
    counts[CPU_COLUMN] = int(sum(valid_cpu(raw).sum() for _, raw in source_series))
    return {"status": "SOURCE_ONLY_NORMALIZATION_PARAMETERS", "fit_role": "source20:first86400seconds",
            "raw_metric_columns": RAW_COLUMNS, "non_CPU_column_indices_0based": list(NON_CPU_COLUMNS),
            "log1p_q99_scales_by_raw_column": scales, "source_observed_valid_counts": counts,
            "quantile_method": "linear", "scale_floor": 1e-6,
            "non_CPU_transform": "log1p(finite nonnegative raw value)/source pooled log1p q99; no clipping",
            "CPU_and_CPU_mean_and_CPU_diff_and_labels": "recorded value /100; diff is signed; no clipping",
            "model_training_or_predictions": False}


def windows(timestamps, raw_metrics, normalization, horizon, origins=None):
    origins = complete_origins(raw_metrics, horizon) if origins is None else np.asarray(origins, np.int64)
    features = causal_features(timestamps, raw_metrics, normalization)
    x = np.stack([features[o-L+1:o+1] for o in origins]).astype(np.float32)
    cpu = np.asarray(raw_metrics)[:, CPU_COLUMN]
    y = np.stack([cpu[o+1:o+int(horizon)+1]/100.0 for o in origins]).astype(np.float32)
    if x.shape != (len(origins), L, 25) or y.shape != (len(origins), int(horizon)):
        raise RuntimeError("Window shape mismatch")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise RuntimeError("Invalid raw true label was requested")
    return x, y

