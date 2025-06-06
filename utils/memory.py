# ── utils.memory ──────────────────────────────────────────────────────────
import os
try:
    import psutil
except ImportError:
    psutil = None


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

def compute_max_threads(memory_limited_workers: int, max_workers: int, task_count: int = 5) -> tuple[int, int]:
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
        max_threads = min(task_count, max_threads)
        return max_threads, max_workers
    elif max_workers >= cpu_count / 2:
        max_threads = min(task_count, memory_limited_workers)
        adjusted_max_workers = min(max(1, memory_limited_workers // max_threads), max_workers)
        return max_threads, adjusted_max_workers
    else:
        max_threads = 2
        max_workers = min(max(1, int(memory_limited_workers // max_threads)), max_workers)
        return max_threads, max_workers