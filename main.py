"""Command‑line entry point.

Example:
    python -m eeg_pipeline.main process \                --data‑dir /path/to/mat_files \                --output‑dir /path/to/output
"""

from pathlib import Path
import json
import numpy as np
import typer

from config import *
from logger import get_logger
from utils.fileio import load_subject_mat
from preprocessing.channel_correction import cz_interpolation, reorder_channels
from preprocessing.segmentation import segment_signal
from features.frequency import bandpower
from features.nonlinear import (
    sample_entropy,
    permutation_entropy,
    higuchi_fd,
    lyapunov_exponent,
)

app = typer.Typer()
logger = get_logger("main")


@app.command(help="List all variables inside each S*.mat file for quick debugging.")
def inspect(
    data_dir: Path = typer.Option(
        None,
        help="Folder that contains S*.mat files (defaults to storage/input_data).",
    )
) -> None:
    """
    Iterate over every `S*.mat` file and print variable names, shapes, and types.

    The command first tries `scipy.io.loadmat()` (MAT-file ≤ v7).
    If the file is v7.3 (HDF5) it falls back to `mat73.loadmat()` automatically.
    Results are logged to both the console and *storage/logs/inspect.log*.
    """
    from config import INPUT_DIR
    from logger import get_logger
    import mat73
    from scipy.io import loadmat

    logger = get_logger("inspect")

    # Use default directory when no --data-dir is provided
    if data_dir is None:
        data_dir = INPUT_DIR

    logger.info(f"Scanning MAT files in: {data_dir}")

    # Glob all S*.mat files
    for mat_file in sorted(data_dir.glob("S*.mat")):
        logger.info(f"--- {mat_file.name} ---")

        # Try classic MAT reader first, then fall back to mat73
        try:
            mat_dict = loadmat(mat_file)
        except NotImplementedError:
            logger.info("MAT-file v7.3 detected – switching to mat73.")
            mat_dict = mat73.loadmat(mat_file)

        # List every variable except MATLAB’s meta‐fields
        for key, value in mat_dict.items():
            if key.startswith("__"):
                continue
            shape = getattr(value, "shape", "")
            logger.info(f"{key:<20}  type={type(value).__name__:<15}  shape={shape}")

@app.command()
def process(data_dir: Path = None, output_dir: Path = None, fs: int = FS):
    """Process all S*.mat files and extract features."""
    from config import INPUT_DIR, OUTPUT_DIR, IMAGE_DIR
    if data_dir is None:
        data_dir = INPUT_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
    subject_files = sorted(data_dir.glob("S*.mat"))
    logger.info(f"Found {len(subject_files)} subject files.")

    for mat_file in subject_files:
        subj = mat_file.stem
        mat_dict = load_subject_mat(mat_file)
        subj_results = {}

        for task, key_list in TASK_MAP.items():
            # Gather all trials belonging to this task
            trial_signals = []
            for key in key_list:
                if key not in mat_dict:
                    logger.warning(f"{key} missing in {mat_file.name}; skipping.")
                    continue
                trial_signals.append(mat_dict[key])

            if not trial_signals:
                logger.warning(f"No data for {task} in {mat_file.name}; skipping task.")
                continue

            # Concatenate trials in time: shape -> (channels, total_samples)
            sig = np.concatenate(trial_signals, axis=1)
            sig = reorder_channels(sig)
            sig = cz_interpolation(sig)

            win_len = fs * 1            # 1‑second window
            overlap = int(win_len * 0.4)
            segments = segment_signal(sig, win_len, overlap)  # (wins × ch × samples)
            logger.debug(
                f"{subj} – {task}: win_len={win_len} samples, "
                f"overlap={overlap} samples, n_windows={len(segments)}"
            )
            win_feats = []
            for seg in segments:
                ch_feats = []
                for ch in seg:
                    bp_delta = bandpower(ch, fs, (0.1, 4))
                    se = sample_entropy(ch)
                    pe = permutation_entropy(ch)
                    fd = higuchi_fd(ch)
                    ly = lyapunov_exponent(ch)
                    ch_feats.append([bp_delta, se, pe, fd, ly])
                win_feats.append(np.mean(ch_feats, axis=0))

            subj_results[task] = np.mean(win_feats, axis=0).tolist()

        out_file = output_dir / f"{subj}_features.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(subj_results, f, indent=2)
        logger.info(f"Saved {out_file.name}")


if __name__ == "__main__":
    app()
