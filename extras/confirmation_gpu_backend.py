"""CUDA float64 learner backend for the locally frozen X11c design.

Search, RNG, repairs, folds, score thresholds, metrics, and checkpoint selection
reuse the preserved NumPy implementation.  Ridge linear algebra and kNN distance
matrices execute on CUDA.  Inner-fold scaler statistics are calculated with the
original NumPy mean/std rules, then subtraction/division run on CUDA.  NumPy
argpartition is deliberately retained for the original kNN ordering convention.

CUDA and CPU floating-point reductions/solvers can differ.  This backend is a
separate execution version, not a claim of bitwise replay of CPU search paths.
There is no CPU learner fallback when CUDA is unavailable.
"""
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "extras"))

import numpy as np
import torch

from efsbench import algos as A, learners as L, metrics as M, protocols as P, splits as S
from efsbench.wrapper import Wrapper
from ex_suites6 import CountedRun

CPS = [250, 1000, 4000]
DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
BACKEND_VERSION = "x11c_cuda_float64_v1"
_READY = False
_COUNTS = {}


def reset_diagnostics():
    """Reset operation counters at the start of one job (no learner fitting)."""
    _COUNTS.clear()
    _COUNTS.update(cuda_ridge_fits=0, cuda_knn_distance_matrices=0,
                   cuda_scaling_transforms=0, cpu_knn_argpartitions=0,
                   cuda_primal_solves=0, cuda_dual_solves=0,
                   max_matrix_rows=0, max_selected_features=0,
                   cuda_dispatches=0)


reset_diagnostics()


def require_cuda():
    global _READY
    if not torch.cuda.is_available():
        raise RuntimeError("X11c CUDA backend requires an available CUDA device; CPU fallback is forbidden")
    if not _READY:
        torch.cuda.set_device(DEVICE)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        _READY = True
    return DEVICE


def synchronize():
    torch.cuda.synchronize(require_cuda())


def device_metadata():
    """Inspect device/runtime; does not load any benchmark or fit a learner."""
    device = require_cuda()
    props = torch.cuda.get_device_properties(device)
    return dict(backend_version=BACKEND_VERSION, torch_version=str(torch.__version__),
                torch_cuda_version=str(torch.version.cuda), device=str(device),
                device_name=props.name, device_total_memory_bytes=int(props.total_memory),
                compute_capability=[int(props.major), int(props.minor)],
                learner_dtype="float64",
                tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
                cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                dispatch_counter_unit="CUDA learner/scaling operations, not kernel launches",
                gpu_operations=["inner-fold feature scaling", "ridge centering and weighted linear algebra",
                                "ridge primal/dual solve", "kNN squared-distance matrix"],
                cpu_operations=["NumPy training-only mean/std scaler statistics", "NumPy search and RNG",
                                "NumPy kNN argpartition", "NumPy/scipy metrics"],
                cpu_learner_fallback=False, bitwise_cpu_equivalence=False)


def diagnostics():
    return dict(_COUNTS, device=str(DEVICE), learner_dtype="float64",
                backend_version=BACKEND_VERSION)


def _float_tensor(array):
    require_cuda()
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np.float64), device=DEVICE, dtype=DTYPE)


def _index_tensor(array):
    require_cuda()
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np.int64), device=DEVICE, dtype=torch.long)


def _ridge_fit_cuda(Xtr, ttr, w, lam):
    """Original class-balanced weighted ridge, including the primal/dual switch."""
    if Xtr.device.type != "cuda" or Xtr.dtype != DTYPE:
        raise RuntimeError("Ridge inputs must be CUDA float64")
    _COUNTS["cuda_ridge_fits"] += 1
    _COUNTS["cuda_dispatches"] += 1
    sw = w.sum()
    mu = (w @ Xtr) / sw
    mt = (w @ ttr) / sw
    Xc, tc = Xtr - mu, ttr - mt
    m, n = int(Xc.shape[1]), int(Xc.shape[0])
    if m <= n:
        Xw = Xc * w[:, None]
        matrix = Xc.T @ Xw
        matrix.diagonal().add_(float(lam))
        beta = torch.linalg.solve(matrix, Xw.T @ tc)
        _COUNTS["cuda_primal_solves"] += 1
    else:
        r = torch.sqrt(w)
        Z = Xc * r[:, None]
        matrix = Z @ Z.T
        matrix.diagonal().add_(float(lam))
        beta = Z.T @ torch.linalg.solve(matrix, r * tc)
        _COUNTS["cuda_dual_solves"] += 1
    return mu, mt, beta


def _fit_predict_tensors(Xtr, ytr_cpu, Xte, spec, target=None, weights=None):
    require_cuda()
    if (Xtr.device.type != "cuda" or Xte.device.type != "cuda"
            or Xtr.dtype != DTYPE or Xte.dtype != DTYPE):
        raise RuntimeError("Learner inputs must be CUDA float64")
    _COUNTS["max_matrix_rows"] = max(_COUNTS["max_matrix_rows"], int(Xtr.shape[0]), int(Xte.shape[0]))
    _COUNTS["max_selected_features"] = max(_COUNTS["max_selected_features"], int(Xtr.shape[1]))
    if spec[0] == "knn":
        # Keep the original _sqdist expression and clamp-before-partition rule.
        distances = (Xte * Xte).sum(1)[:, None] + (Xtr * Xtr).sum(1)[None, :] - 2.0 * (Xte @ Xtr.T)
        distances.clamp_min_(0.0)
        _COUNTS["cuda_knn_distance_matrices"] += 1
        _COUNTS["cuda_dispatches"] += 1
        distances_cpu = distances.detach().cpu().numpy()
        nn = np.argpartition(distances_cpu, spec[1] - 1, axis=1)[:, :spec[1]]
        _COUNTS["cpu_knn_argpartitions"] += 1
        return np.asarray(ytr_cpu)[nn].mean(1)
    if spec[0] != "ridge":
        raise ValueError("Unknown preserved learner: " + str(spec[0]))
    if target is None:
        target = _float_tensor(2.0 * np.asarray(ytr_cpu) - 1.0)
    if weights is None:
        weights = _float_tensor(L.balanced_weights(np.asarray(ytr_cpu)))
    mu, mt, beta = _ridge_fit_cuda(Xtr, target, weights, spec[1])
    return ((Xte - mu) @ beta + mt).detach().cpu().numpy()


@torch.no_grad()
def fit_predict_gpu(Xtr, ytr, Xte, spec):
    """Drop-in real-valued scorer; all expensive learner algebra uses CUDA."""
    return _fit_predict_tensors(_float_tensor(Xtr), np.asarray(ytr), _float_tensor(Xte), spec)


class CudaFoldScaledWrapper(Wrapper):
    """Cache raw predictors on CUDA; fit scaler statistics on each inner train fold."""

    def __init__(self, Xraw, y, folds, spec, metric="ba"):
        require_cuda()
        self.Xraw, self.y, self.folds, self.spec, self.metric = Xraw, np.asarray(y), folds, spec, metric
        self.thr = L.threshold(spec)
        self.calls, self.cache, self.p = 0, {}, Xraw.shape[1]
        self.Xgpu = _float_tensor(Xraw)
        self.statistics, self.gpu_folds = [], []
        for tr, te in folds:
            train = Xraw[tr]
            mu, sd = train.mean(0), train.std(0)
            sd = np.where(sd > 0, sd, 1.0)
            self.statistics.append((mu, sd))
            fold = dict(train_index=_index_tensor(tr), test_index=_index_tensor(te),
                        mean=_float_tensor(mu), std=_float_tensor(sd), ytrain=self.y[tr])
            if spec[0] == "ridge":
                fold.update(target=_float_tensor(2.0 * self.y[tr] - 1.0),
                            weights=_float_tensor(L.balanced_weights(self.y[tr])))
            self.gpu_folds.append(fold)

    @torch.no_grad()
    def scores(self, mask):
        idx = np.flatnonzero(mask)
        index_gpu = _index_tensor(idx)
        selected = self.Xgpu.index_select(1, index_gpu)
        scores = np.empty(len(self.y))
        for (_, te), fold in zip(self.folds, self.gpu_folds):
            mu = fold["mean"].index_select(0, index_gpu)
            sd = fold["std"].index_select(0, index_gpu)
            train = (selected.index_select(0, fold["train_index"]) - mu) / sd
            valid = (selected.index_select(0, fold["test_index"]) - mu) / sd
            _COUNTS["cuda_scaling_transforms"] += 2
            _COUNTS["cuda_dispatches"] += 2
            scores[te] = _fit_predict_tensors(train, fold["ytrain"], valid, self.spec,
                                             fold.get("target"), fold.get("weights"))
        return scores

    def evaluate(self, mask):
        self.calls += 1
        idx = np.flatnonzero(mask)
        key = idx.astype(np.int32).tobytes()
        if key not in self.cache:
            self.cache[key] = self._err_from_scores(self.scores(mask))
        return self.cache[key]

    def oof_metrics(self, mask):
        scores = self.scores(mask)
        return M.all_metrics(self.y, (scores > self.thr).astype(np.int64), scores)


@torch.no_grad()
def strict_core_gpu(Xtr, ytr, Xte, yte, algo, learner, inner_key, algo_key, checkpoints=CPS):
    """Preserved strict core with a versioned CUDA learner backend.

    No outer-test prediction is computed until the full search has completed.
    Return schema is identical to ex_suites6.strict_core.
    """
    require_cuda()
    inner = S.make_folds(ytr, 5, S.rng_for(*inner_key))
    spec = L.SPECS[learner]
    wrapper = CudaFoldScaledWrapper(Xtr, ytr, inner, spec)
    run = CountedRun(wrapper, checkpoints)
    A.run_algo(algo, run, S.rng_for(*algo_key))
    # Outer-only scaler fitting remains the original NumPy implementation.
    train_scaled, test_scaled = P.standardize(Xtr, Xte)
    train_gpu, test_gpu = _float_tensor(train_scaled), _float_tensor(test_scaled)
    ytr = np.asarray(ytr)
    target = _float_tensor(2.0 * ytr - 1.0) if spec[0] == "ridge" else None
    weights = _float_tensor(L.balanced_weights(ytr)) if spec[0] == "ridge" else None
    out = {}
    for cp in checkpoints:
        snap = run.snap[cp]
        idx, err = P._sel(algo, snap)
        index_gpu = _index_tensor(idx)
        scores = _fit_predict_tensors(train_gpu.index_select(1, index_gpu), ytr,
                                      test_gpu.index_select(1, index_gpu), spec, target, weights)
        metrics = M.all_metrics(yte, (scores > L.threshold(spec)).astype(np.int64), scores)
        out[str(cp)] = dict(k=int(len(idx)), inner_err=float(err), sel=[int(i) for i in idx],
                            calls=snap["calls"], distinct=snap["distinct"],
                            **{"test_" + key: float(value) for key, value in metrics.items()})
    synchronize()
    return out, [{"train": tr.tolist(), "validation": te.tolist()} for tr, te in inner]
