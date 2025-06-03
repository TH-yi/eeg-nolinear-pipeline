# ── utils.memory ──────────────────────────────────────────────────────────
import os
try:
    import psutil
except ImportError:
    psutil = None


def safe_worker_count(
    bytes_per_job: int,
    max_cpu: int | None = None,
    safety_ratio: float = 0.6,
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
    print(f"safe worker count: {safe_worker_count}")
    return safe_worker_count
