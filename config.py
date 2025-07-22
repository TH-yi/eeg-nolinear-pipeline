from pathlib import Path

TASK_MAP: dict[str, list[str]] = {
    # "RST1": ["Sig_RST1"],
    # "RST2": ["Sig_RST2"],
    "IDG": ["Sig_IDG_1", "Sig_IDG_2", "Sig_IDG_3"],
    "IDE": ["Sig_IDE_1", "Sig_IDE_2", "Sig_IDE_3"],
    "IDR": ["Sig_IDR_1", "Sig_IDR_2", "Sig_IDR_3"],

}
# Project root = .../eeg_pipeline_project
ROOT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = ROOT_DIR / "storage"

INPUT_DIR = STORAGE_DIR / "input_data"
OUTPUT_DIR = STORAGE_DIR / "output_data"
IMAGE_DIR = STORAGE_DIR / "images"
LOGGER_DIR = STORAGE_DIR / "logs"

FS = 500
CHANNELS = 64
TAU = 10  # for f7 and f8 tau parameter
LAG = 1  # for f1 and f3 lag(tau) parameter
EMB_DIM = 2

# Default number of parallel tasks per feature channel
PARALLEL_TASK_COUNT = 3

# Memory limit threshold (adjust based on empirical observations)
MAX_WORKER_MEMORY_LIMIT = 0.001 # Minimum memory required per worker (unit depends on implementation)

# CPU utilization ratio (used in functions like safe_worker_count)
CPU_UTILIZATION_RATIO = 0.001  # Used to estimate safe_worker_count
