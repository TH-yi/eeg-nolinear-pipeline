# -*- coding: utf-8 -*-
"""crqa_py — Cross-Recurrence Quantification Analysis in pure Python
===================================================================
*Numerically identical* to Norbert Marwan’s MATLAB `crqa.m` for all 13
indices, plus a wrapper that mimics `nonlinear_analysis()` I/O.

▸ GPU optional (CuPy) — falls back to NumPy when unavailable.
▸ now supports **CPU ProcessPool + single-GPU worker** mode
  (set `use_gpu=True` and `max_workers>1` in `rqa_analysis`).
"""
from __future__ import annotations
from utils.timer import timer
from utils.memory import wait_for_available_memory, wait_for_available_gpu_memory

import itertools, uuid, warnings, queue as _pyqueue
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, managers

try:
    import cupy as cp          # GPU optional
except ModuleNotFoundError:
    cp = None

Array  = Union[np.ndarray, "cp.ndarray"]
_DTYPE = np.float64

###############################################################################
# Helper utilities
###############################################################################
def _backend(x: Array):
    """Return cp or np backend matching `x`."""
    return cp if (cp is not None and isinstance(x, cp.ndarray)) else np

def _to_ndarray(x: Array, use_gpu: bool = True) -> Array:
    xp = cp if (use_gpu and cp) else np
    arr = xp.asarray(x, dtype=_DTYPE)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr

def _zscore(x: Array) -> Array:
    return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, ddof=0, keepdims=True)

def _embed(ts: Array, m: int, tau: int) -> Array:
    n, dim = ts.shape
    if m == 1:
        return ts.copy()
    rows = n - (m - 1) * tau
    if rows < 1:
        raise ValueError("Too few samples for embedding.")
    xp = _backend(ts)
    emb = xp.empty((rows, dim * m), dtype=_DTYPE)
    for i in range(m):
        emb[:, i * dim:(i + 1) * dim] = ts[i * tau:i * tau + rows]
    return emb

###############################################################################
# Distance & RP
###############################################################################
_METHODS: Dict[str, Literal["max", "eu", "min", "nr"]] = {
    "max": "max", "euclidean": "eu", "eu": "eu", "min": "min", "nr": "nr",
}

@timer()
def _distance_matrix(x: Array, y: Array, method: str, use_gpu: bool) -> Array:
    xp = cp if (use_gpu and cp) else np

    n = x.shape[0]
    size_gb = (n * n * 8) / 1024 ** 3  # float64
    wait_for_available_memory(size_gb)

    if method == "max":     # Chebyshev
        return xp.abs(x[:, None] - y).max(axis=2)
    if method == "min":     # Manhattan
        return xp.abs(x[:, None] - y).sum(axis=2)
    if method == "nr":      # normalised Euclidean
        nx = xp.linalg.norm(x, axis=1, keepdims=True); ny = xp.linalg.norm(y, axis=1, keepdims=True)
        nx[nx == 0.0] = 1.0; ny[ny == 0.0] = 1.0
        x, y = x / nx, y / ny
    diff = x[:, None] - y
    return xp.sqrt((diff ** 2).sum(axis=2))

def _rp_binary(dmat: Array, eps: float) -> Array:
    xp = _backend(dmat)
    return (dmat <= eps).astype(_DTYPE)

###############################################################################
# Line statistics (diag / vert / white)
###############################################################################
@timer()
def _diag_line_lengths(rp: Array, lmin: int):
    xp, n, lens = _backend(rp), rp.shape[0], []
    for k in range(-n + 1, n):
        if k == 0:
            continue
        d = xp.diag(rp, k=k)
        idx = xp.where(d == 1)[0]
        if idx.size == 0:
            continue
        splits = xp.where(xp.diff(idx) != 1)[0] + 1
        runs = xp.split(idx, splits.tolist())
        lens.extend(int(len(r)) for r in runs if len(r) >= lmin)
    arr = xp.asarray(lens, dtype=_DTYPE)
    return (arr.get() if xp is cp else arr), n

@timer()
def _vertical_line_lengths(rp: Array, vmin: int):
    xp, n, lens = _backend(rp), rp.shape[1], []
    for col in range(n):
        idx = xp.where(rp[:, col] == 1)[0]
        if idx.size == 0:
            continue
        splits = xp.where(xp.diff(idx) != 1)[0] + 1
        runs = xp.split(idx, splits.tolist())
        lens.extend(int(len(r)) for r in runs if len(r) >= vmin)
    arr = xp.asarray(lens, dtype=_DTYPE)
    return arr.get() if xp is cp else arr

@timer()
def _white_vertical_lengths(rp: Array):
    xp, n, lens, rp_w = _backend(rp), rp.shape[1], [], 1 - rp
    for col in range(n):
        idx = xp.where(rp_w[:, col] == 1)[0]
        if idx.size == 0:
            continue
        splits = xp.where(xp.diff(idx) != 1)[0] + 1
        runs = xp.split(idx, splits.tolist())
        lens.extend(int(len(r)) for r in runs)
    arr = xp.asarray(lens, dtype=_DTYPE)
    return arr.get() if xp is cp else arr

@timer()
def _recurrence_times(rp: Array) -> Tuple[float, float]:
    xp, n = _backend(rp), rp.shape[1]
    t1, t2 = [], []
    for col in range(n):
        col_vec = rp[:, col]
        rps = xp.where(col_vec == 1)[0]
        if rps.size >= 2:
            t1.extend(xp.diff(rps).tolist())
        rps2 = xp.where(xp.diff(col_vec.astype(int)) == 1)[0]
        if rps2.size >= 2:
            t2.extend(xp.diff(rps2).tolist())
    return float(np.mean(t1)) if t1 else np.nan, float(np.mean(t2)) if t2 else np.nan

def _entropy(counts: np.ndarray) -> float:
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    nz = p > 0
    return -(p[nz] * np.log2(p[nz])).sum() * np.log(2.0)

###############################################################################
# Network measures
###############################################################################
def is_symmetric_matrix(A: Array, atol: float = 0.0) -> bool:
    xp = _backend(A)
    return xp.allclose(A, A.T, atol=atol)

def compute_triangles_fast(A: Array):
    xp = _backend(A)
    A = A.astype(xp.int8 if xp is np else xp.int32, copy=False)
    A2 = A @ A
    tri = (A2 * A).sum(axis=1)
    return tri, float(tri.sum()), float(A2.sum())

def compute_triangles_with_sym_check(A: Array):
    xp = _backend(A)
    if is_symmetric_matrix(A):
        return compute_triangles_fast(A)
    tri = xp.einsum("ij,jk,ki->i", A, A, A)
    return tri, float(xp.einsum("ij,jk,ki->", A, A, A)), float((A @ A).sum())

def _network_measures(rp: Array):
    xp = _backend(rp)
    A = rp.astype(int, copy=False)
    kv = A.sum(axis=1)
    tri, trace_all, denom = compute_triangles_with_sym_check(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl_local = tri / (kv * (kv - 1))
    clust = float(xp.nanmean(cl_local))
    trans = float((trace_all / denom) if denom > 0 else xp.nan)
    return clust, trans

###############################################################################
# CRQA core (network-less path)
###############################################################################
@dataclass
class CRQAResult:
    RR: float; DET: float; L_mean: float; L_max: float; ENTR: float; LAM: float
    TT: float; V_max: float; T1: float; T2: float; RTE: float
    Clust: float; Trans: float

    def as_array(self) -> np.ndarray:
        return np.array([self.RR, self.DET, self.L_mean, self.L_max, self.ENTR,
                         self.LAM, self.TT, self.V_max, self.T1, self.T2,
                         self.RTE, self.Clust, self.Trans], dtype=_DTYPE)

@timer()
def _crqa_no_net(
    x: Array, *, m: int, tau: int, e: float, lmin: int, vmin: int,
    theiler: int, method: str, normalize: bool, use_gpu: bool,
    max_threads_per_channel: int = 1
) -> Tuple[CRQAResult, Array]:

    x_arr = _to_ndarray(x, use_gpu)
    if normalize:
        x_arr = _zscore(x_arr)

    x_emb = _embed(x_arr, m, tau)
    method_key = _METHODS.get(method.lower(), method.lower())
    if method_key not in _METHODS.values():
        raise NotImplementedError(f"method '{method}' not supported")

    dmat = _distance_matrix(x_emb, x_emb, method_key, use_gpu)
    rp = (dmat <= e).astype(np.bool_)
    del dmat
    if use_gpu and cp:
        cp._default_memory_pool.free_all_blocks()

    if theiler > 0:
        xp, n = _backend(rp), rp.shape[0]
        for k in range(-theiler, theiler + 1):
            diag_idx = xp.arange(max(0, -k), min(n, n - k))
            rp[diag_idx, diag_idx + k] = 0

    rp_cpu = rp.get() if (cp is not None and isinstance(rp, cp.ndarray)) else rp
    N_all = rp_cpu.size - (0 if theiler == 0 else
                           2 * rp_cpu.shape[0] * theiler - theiler * (theiler + 1))
    RR = rp.sum() / N_all
    # ---------------- Parallel phase ----------------
    results = {}

    def compute_diag():
        lens, _ = _diag_line_lengths(rp, lmin)
        return {
            'L_max': float(lens.max()) if lens.size else 0.0,
            'L_mean': float(lens.mean()) if lens.size else np.nan,
            'DET': lens.sum() / rp_cpu.sum() if rp_cpu.sum() else np.nan,
            'ENTR': _entropy(np.histogram(lens, bins=np.arange(1, rp_cpu.shape[0] + 1))[0])
        }

    def compute_vert():
        lens = _vertical_line_lengths(rp, vmin)
        return {
            'V_max': float(lens.max()) if lens.size else 0.0,
            'TT': float(lens.mean()) if lens.size else np.nan,
            'LAM': lens.sum() / rp_cpu.sum() if rp_cpu.sum() else np.nan
        }

    def compute_white():
        white = _white_vertical_lengths(rp)
        if white.size and white.max() > 0:
            rte = _entropy(np.histogram(white, bins=np.arange(1, white.max() + 2))[0]) / np.log(white.max())
        else:
            rte = np.nan
        return {'RTE': rte}

    def compute_rec_times():
        t1, t2 = _recurrence_times(rp)
        return {'T1': t1, 'T2': t2}

    tasks = {
        'diag': compute_diag,
        'vert': compute_vert,
        'white': compute_white,
        'recur': compute_rec_times
    }

    if max_threads_per_channel > 1:
        with ThreadPoolExecutor(max_threads_per_channel) as ex:
            future_to_key = {ex.submit(func): key for key, func in tasks.items()}
            for f in as_completed(future_to_key):
                key = future_to_key[f]
                results.update(f.result())
    else:
        # fallback to sequential
        for key, func in tasks.items():
            results.update(func())

    res = CRQAResult(
        RR=RR,
        DET=results['DET'],
        L_mean=results['L_mean'],
        L_max=results['L_max'],
        ENTR=results['ENTR'],
        LAM=results['LAM'],
        TT=results['TT'],
        V_max=results['V_max'],
        T1=results['T1'],
        T2=results['T2'],
        RTE=results['RTE'],
        Clust=np.nan,
        Trans=np.nan
    )
    return res, rp_cpu

###############################################################################
# ----------------------  GPU worker process  --------------------------- ###
###############################################################################
def _gpu_worker(task_q: managers.QueueProxy,
                result_q: managers.QueueProxy,
                device_id: int = 0):
    """GPU process: fetch RP from shared memory, compute network measures."""
    if cp is None:
        result_q.put(("ERROR", "CuPy not available"))
        return

    cp.cuda.Device(device_id).use()

    while True:
        try:
            job = task_q.get(timeout=1)
        except _pyqueue.Empty:
            continue

        if job == "STOP":
            break

        try:
            qlen = task_q.qsize()
        except (AttributeError, NotImplementedError):
            qlen = "N/A"
        #print(f"[GPU Worker] got job, pending: {qlen}")

        job_id, shm_name, shape, dtype_str = job
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            rp_np = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)

            # Estimate GPU memory usage
            required_bytes = rp_np.nbytes
            if not wait_for_available_gpu_memory(required_bytes):
                raise MemoryError(f"Not enough GPU memory for {job_id}, needs {required_bytes / 1024 ** 2:.2f} MB")

            rp_gpu = cp.asarray(rp_np)  # H→D zero-copy
            clust, trans = _network_measures(rp_gpu)
            del rp_np
            del rp_gpu
            cp._default_memory_pool.free_all_blocks()
            result_q.put((job_id, clust, trans))
        except Exception as exc:
            result_q.put((job_id, "ERROR", repr(exc)))
            raise
        finally:
            shm.close()
        try:
            qlen = task_q.qsize()
        except (AttributeError, NotImplementedError):
            qlen = "N/A"
        #print(f"[GPU Worker] finished job, pending: {qlen}")

###############################################################################
# CPU-side worker (process) that offloads network part ------------------- ###
###############################################################################
@timer()
def _cpu_rqa_worker(
    sig: np.ndarray, ch_name: str, cfg: dict,
    task_q: Optional[managers.QueueProxy],
    result_q: Optional[managers.QueueProxy]
) -> np.ndarray:
    """Run CRQA up to RP; offload network measures to GPU via queues."""
    try:
        res_base, rp = _crqa_no_net(sig, **cfg)         # heavy part

        # ---------- network measures ----------
        if task_q is None:          # no GPU → local CPU
            clust, trans = _network_measures(rp)
        else:                       # send to GPU
            job_id = uuid.uuid4().hex
            shm = shared_memory.SharedMemory(create=True, size=rp.nbytes)
            shm_arr = np.ndarray(rp.shape, dtype=rp.dtype, buffer=shm.buf)
            shm_arr[...] = rp
            task_q.put((job_id, shm.name, rp.shape, rp.dtype.str))

            # wait result
            while True:
                jid, *data = result_q.get()
                if jid == job_id:
                    if data[0] == "ERROR":
                        raise RuntimeError(f"GPU worker error: {data[1]}")
                    clust, trans = data
                    break
            shm.unlink()

        result = res_base.as_array()
        result[-2] = clust
        result[-1] = trans
        return result
    except Exception as exc:
        warnings.warn(f"[{ch_name}] failed: {exc}")
        raise

###############################################################################
# Public batch API
###############################################################################
@timer()
def rqa_analysis(
    signal2: np.ndarray, *, tau: int = 1, emb_dim: Optional[int] = None,
    m: Optional[int] = None, e: float = 0.1, lmin: int = 2, vmin: int = 2,
    theiler: int = 1, method: str = "max", normalize: bool = True,
    flatten: bool = True, tqdm_progress=None, use_gpu: bool = True,
    fs: float = 500.0, max_workers: Optional[int] = None,
    max_threads_per_channel: int = 1,
    **kwargs
) -> np.ndarray:
    if signal2.ndim != 2:
        raise ValueError("`signal2` must be 2-D (channels × samples)")

    n_ch, _ = signal2.shape
    m_final = m or emb_dim or 1
    cfg_common = dict(m=m_final, tau=tau, e=e, lmin=lmin, vmin=vmin,
                      theiler=theiler, method=method,
                      normalize=normalize, use_gpu=use_gpu, max_threads_per_channel=max_threads_per_channel)

    # --- decide strategy ---
    # thread_only = (not max_workers or max_workers <= 1) or not use_gpu
    thread_only = not use_gpu
    results: list[np.ndarray] = []

    if thread_only:
        for i in range(n_ch):
            res = _cpu_rqa_worker(signal2[i], f"Ch{i+1}", cfg_common,
                                  None, None)
            results.append(res)
            if tqdm_progress is not None:
                tqdm_progress.update()
        feat = np.vstack(results)
        return feat.ravel() if flatten else feat

    # -------- CPU ProcessPool + GPU worker --------
    cpu_workers = max(max_workers - 1, 1)
    manager = mp.Manager()                    # creates shared server process
    task_q: managers.QueueProxy   = manager.Queue()
    result_q: managers.QueueProxy = manager.Queue()

    gpu_proc = mp.Process(target=_gpu_worker,
                          args=(task_q, result_q, 0),
                          daemon=True)
    gpu_proc.start()

    with ProcessPoolExecutor(max_workers=cpu_workers,
                             mp_context=mp.get_context('spawn')) as ex:
        futures = [ex.submit(_cpu_rqa_worker,
                             signal2[i], f"Ch{i+1}", cfg_common,
                             task_q, result_q)
                   for i in range(n_ch)]
        for f in as_completed(futures):
            results.append(f.result())
            if tqdm_progress is not None:
                tqdm_progress.update()

    task_q.put("STOP")
    gpu_proc.join()

    feat = np.vstack(results)
    return feat.ravel() if flatten else feat


__all__ = ["rqa_analysis", "CRQAResult"]
