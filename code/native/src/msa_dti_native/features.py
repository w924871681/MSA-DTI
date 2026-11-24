import numpy as np
import hashlib,json
CPU_COLUMN=3
NON_CPU_COLUMNS=[0,1,2,4,5,6,7,8,9]
L=96
HORIZONS=(1,4)
SUPPORT_K=(10,20)

def causal_features(timestamps, raw_metrics, normalization, return_dependencies=False):
    """Reset state at this raw role boundary; only current/past records used.

    Future mutation or truncation cannot change an existing prefix. Invalid raw
    values have mask0 and carry the previous valid value, with initial value0.
    CPU >100 is a valid recorded value. The derived mean uses the last <=5 filled
    CPU values; its mask requires all corresponding raw CPU records valid.
    """
    t = np.asarray(timestamps, dtype=np.int64)
    raw = np.asarray(raw_metrics, dtype=np.float64)
    if raw.shape != (len(t), 10) or len(t) == 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Bad raw role shape/native timestamp order")
    observed = np.isfinite(raw) & (raw >= 0)
    filled = np.zeros_like(raw)
    deps = np.full(raw.shape, -1, dtype=np.int64)
    last = np.zeros(10, dtype=np.float64)
    last_dep = np.full(10, -1, dtype=np.int64)
    for i in range(len(t)):
        last[observed[i]] = raw[i, observed[i]]
        last_dep[observed[i]] = i
        filled[i] = last
        deps[i] = last_dep
    values = np.empty((len(t), 12), dtype=np.float64)
    for col in NON_CPU_COLUMNS:
        values[:, col] = np.log1p(filled[:, col])/normalization["log1p_q99_scales_by_raw_column"][col]
    values[:, CPU_COLUMN] = filled[:, CPU_COLUMN]/100.0
    masks = np.zeros((len(t), 12), dtype=np.float64)
    masks[:, :10] = observed
    all_deps = np.full((len(t), 25), -1, dtype=np.int64)
    all_deps[:, :10] = deps
    for i in range(len(t)):
        lo = max(0, i-4)
        values[i, 10] = filled[lo:i+1, CPU_COLUMN].mean()/100.0
        masks[i, 10] = np.all(observed[lo:i+1, CPU_COLUMN])
        all_deps[i, 10] = deps[lo:i+1, CPU_COLUMN].max()
    values[:, 11] = 0.0
    if len(t) > 1:
        values[1:, 11] = np.diff(filled[:, CPU_COLUMN])/100.0
        masks[1:, 11] = observed[1:, CPU_COLUMN] & observed[:-1, CPU_COLUMN]
        all_deps[1:, 11] = np.maximum(deps[1:, CPU_COLUMN], deps[:-1, CPU_COLUMN])
    # Each mask observes validity at current/past record positions, regardless
    # of whether the numeric value is carried from an earlier record.
    all_deps[:, 12:24] = np.arange(len(t))[:, None]
    gap = np.zeros((len(t), 1), dtype=np.float64)
    gap[1:, 0] = np.diff(t)/1200.0
    all_deps[:, 24] = np.arange(len(t))
    features = np.column_stack([values, masks, gap]).astype(np.float32)
    if features.shape != (len(t), 25) or not np.all(np.isfinite(features)):
        raise RuntimeError("Nonfinite or wrong shape causal feature tensor")
    if np.any(all_deps > np.arange(len(t))[:, None]):
        raise RuntimeError("Future dependency in causal feature tensor")
    return (features, all_deps) if return_dependencies else features

def array_sha256(value):
    """Dtype+shape+canonical C bytes, independent of NPZ compression metadata."""
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()

def raw_role_bounds(horizon, support_k):
    """Exact ordinary-size allocation of frozen center_api, .2:.1:.1."""
    h, k = int(horizon), int(support_k)
    if h not in HORIZONS or k not in SUPPORT_K:
        raise ValueError("H/K outside the declared Bitbrains protocol")
    n_sup = L+h+k-1
    remaining = 3000-n_sup
    p = np.asarray([.2, .1, .1], dtype=np.float64)
    raw = p/p.sum()*remaining
    sizes = np.maximum(np.floor(raw).astype(int), 1)
    while int(sizes.sum()) < remaining:
        sizes[int(np.argmax(raw-sizes))] += 1
    if min(sizes) < L+h:
        raise RuntimeError("Cannot allocate the fixed raw role segments")
    bounds = np.cumsum([0, n_sup, *sizes]).tolist()
    return {r: (bounds[i], bounds[i+1]) for i, r in enumerate(("support", "validation", "check", "test"))}
