import functools
import logging
import os
import psutil
from time import perf_counter

def timer(tag: str | None = None, log_gpu: bool = True):
    def _decorator(fn):
        lbl = tag or fn.__qualname__

        @functools.wraps(fn)
        def _wrap(*a, **kw):
            proc = psutil.Process(os.getpid())
            mem0 = proc.memory_info().rss / 1024**2  # MB
            try:
                import cupy as cp
                mem_gpu0 = cp.get_default_memory_pool().used_bytes() / 1024**2
            except Exception:
                mem_gpu0 = None

            t0 = perf_counter()
            out = fn(*a, **kw)
            dt = perf_counter() - t0

            mem1 = proc.memory_info().rss / 1024**2
            mem_cpu_diff = mem1 - mem0

            if log_gpu and mem_gpu0 is not None:
                try:
                    import cupy as cp
                    mem_gpu1 = cp.get_default_memory_pool().used_bytes() / 1024**2
                    mem_gpu_diff = mem_gpu1 - mem_gpu0
                except Exception:
                    mem_gpu_diff = None
            else:
                mem_gpu_diff = None

            msg = f"[TIMER] {lbl:<25} {dt:>8.4f}s | ΔCPU: {mem_cpu_diff:>7.2f} MB"
            if mem_gpu_diff is not None:
                msg += f" | ΔGPU: {mem_gpu_diff:>7.2f} MB"
            logging.info(msg)
            return out

        return _wrap
    return _decorator
