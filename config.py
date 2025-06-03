from pathlib import Path

TASK_MAP: dict[str, list[str]] = {

    "RST1": ["Sig_RST1"],
    "RST2": ["Sig_RST2"],
    "IDR": ["Sig_IDR_1", "Sig_IDR_2", "Sig_IDR_3"],
    "IDG": ["Sig_IDG_1", "Sig_IDG_2", "Sig_IDG_3"],
    "IDE": ["Sig_IDE_1", "Sig_IDE_2", "Sig_IDE_3"],


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
