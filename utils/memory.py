# ── utils.memory ──────────────────────────────────────────────────────────
import os
import logging
import time
import psutil


def safe_worker_count(
    bytes_per_job: int,
    max_cpu: int | None = None,
    safety_ratio: float = 0.002,
) -> int:
    max_cpu = max_cpu or os.cpu_count() or 1

    if psutil is None:
        return max_cpu

    avail = psutil.virtual_memory().available
    budget = int(avail * safety_ratio)

    if bytes_per_job == 0:
        return max_cpu

    mem_limited = max(1, budget // bytes_per_job)
    safe_worker_count = min(mem_limited, max_cpu)
    # print(f"safe worker count: {safe_worker_count}")
    return safe_worker_count

def memory_limited_worker_count(
    bytes_per_job: int,
    safety_ratio: float = 0.002,
) -> int:
    """
    Estimate the maximum number of workers allowed purely based on memory.

    Parameters
    ----------
    bytes_per_job : int
        Estimated memory usage (in bytes) of a single job.
    safety_ratio : float
        Fraction of available memory to use. Default is 0.2%.

    Returns
    -------
    int
        Theoretical max number of workers constrained only by memory.
    """
    if psutil is None:
        raise RuntimeError("psutil is required to estimate memory usage.")

    avail = psutil.virtual_memory().available
    budget = int(avail * safety_ratio)

    if bytes_per_job <= 0:
        raise ValueError("bytes_per_job must be a positive integer.")

    max_workers = max(1, budget // bytes_per_job)
    # print(f"[Memory-limited] Available: {avail} bytes, Budget: {budget} bytes, Max workers: {max_workers}")
    return max_workers

def compute_max_threads(memory_limited_workers: int, max_workers: int, parallel_task_count: int = 5) -> tuple[int, int]:
    """
    Compute optimal number of threads per process and adjusted max_workers based on memory constraints.

    Returns
    -------
    (max_threads_for_features_per_channel, adjusted_max_workers)
    """
    if not memory_limited_workers or memory_limited_workers < max_workers:
        return 1, max_workers

    cpu_count = os.cpu_count()

    if max_workers >= cpu_count - 2:
        max_threads = max(1, int(1.5*memory_limited_workers) // max_workers)
        max_threads = min(parallel_task_count, max_threads)
        return max_threads, max_workers
    elif max_workers >= cpu_count / 2:
        max_threads = min(parallel_task_count, memory_limited_workers)
        adjusted_max_workers = min(max(1, memory_limited_workers // max_threads), max_workers)
        return max_threads, adjusted_max_workers
    else:
        max_threads = 2
        max_workers = min(max(1, int(memory_limited_workers // max_threads)), max_workers)
        return max_threads, max_workers


def wait_for_available_memory(request_size_gb: float, safety_factor: float = 1.2,
                              check_interval: float = 60, max_wait: float = 3600.0):
    """
    Wait until available RAM is sufficient to allocate a block of size request_size_gb.
    - safety_factor: extra multiplier to prevent tight fit
    - check_interval: seconds between checks
    - max_wait: timeout in seconds (raise RuntimeError if exceeded)
    """
    required = request_size_gb * safety_factor * 1024 ** 3
    waited = 0.0
    while psutil.virtual_memory().available < required:
        logging.warning(f"[MEMGUARD] Waiting for ≥{request_size_gb:.2f} GB free RAM "
                        f"(currently: {psutil.virtual_memory().available / 1024**3:.2f} GB)")
        time.sleep(check_interval)
        waited += check_interval
        if waited >= max_wait:
            raise RuntimeError(f"Timeout waiting for {request_size_gb:.2f} GB free RAM.")


def wait_for_available_gpu_memory(min_required_bytes, device_id=0, interval=60, max_wait=3600):
    import time
    import cupy as cp

    waited = 0
    while True:
        free_bytes, total_bytes = cp.cuda.Device(device_id).mem_info
        if free_bytes >= min_required_bytes:
            return True
        logging.warning(f"[DISPMEMGUARD] Waiting for ≥{min_required_bytes:.2f} GB free DISPRAM "
                        f"(currently: {free_bytes / 1024 ** 3:.2f} GB)")
        time.sleep(interval)
        waited += interval
        if waited >= max_wait:
            raise RuntimeError(f"[GPU Worker] Timeout waiting for {min_required_bytes/1024**2:.2f} MB free GPU memory")

