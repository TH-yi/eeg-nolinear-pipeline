# EEG Nonlinear Feature Extraction Pipeline

This project provides a Python re-implementation of the MATLAB `Nonlinear_Analysis.m` pipeline for extracting 17 (currently the code using 5 of them) nonlinear features from EEG signals on a per-segment, per-channel basis. It supports multiprocessing + multithreading for scalable parallel processing and includes optional diagnostic plotting, memory-aware caching, and integration with `.mat` and `.json` outputs.

## 🔧 Key Features

- **Full parity with MATLAB code** logic for 5 active nonlinear features.
- **17 nonlinear features implemented**, but only f1, f2, f3, f7, and f8 are actively used.
- **Parallelization** using `ProcessPoolExecutor` + `ThreadPoolExecutor`.
- **Automatic memory-aware fallback to disk cache** for large trials.
- **Configurable via `config.py`** for CPU/memory limits and embedding parameters.

## 🚀 Usage

Run the pipeline from the command line using:

```bash
python -m eeg_pipeline.main process \
    --data-dir /path/to/mat_files \
    --output-dir /path/to/output
```

You may also inspect `.mat` structures with:

```bash
python -m eeg_pipeline.main inspect --data-dir /path/to/mat_files
```

## 📁 Directory Structure

```bash
eeg_nolinear_pipeline/
├── main.py
├── config.py
├── storage/
│   ├── input_data/
│   ├── output_data/
│   ├── logs/
```

## ⚙️ Configuration (`config.py`)

| Parameter                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `FS`                      | EEG sampling rate in Hz (default 500)                                       |
| `TAU`                     | Time delay (`tau`) used in entropy & LLE                                    |
| `LAG`                     | Lag (`lag`) used for correlation dimension & LLE                            |
| `EMB_DIM`                 | Embedding dimension (used across many features)                             |
| `PARALLEL_TASK_COUNT`     | Number of concurrent tasks per process (controls thread fan-out)            |
| `MAX_WORKER_MEMORY_LIMIT` | Memory budget per worker (fraction of system RAM)                           |
| `CPU_UTILIZATION_RATIO`   | Fraction of CPU allowed for processing 
......                                     |

## 🧵 Multithreading and Multiprocessing Strategy

- Each **trial** is segmented and processed in parallel.
- Each **segment** is passed to `nonlinear_analysis`, which:
  - Spawns **a pool of processes**, one per EEG channel.
  - Within each process, multiple **threads** compute individual features.

To prevent memory overload:
- If data size is large, the system automatically **saves EEG signals to `.npy` files** and **loads them lazily** via memory-mapping.
- If still out of memory, try to reduce PARALLEL_TASK_COUNT to 1, MAX_WORKER_MEMORY_LIMIT and CPU_UTILIZATION_RATIO to 0.001 or smaller(greater than 0). Such parameters are under config.py.


## 🧪 Active and Legacy Features

The MATLAB reference code used only **five non-linear features**:

| Feature | Description                      |
|---------|----------------------------------|
| f1      | Correlation Dimension            |
| f2      | Higuchi Fractal Dimension        |
| f3      | Largest Lyapunov Exponent        |
| f7      | Sample Entropy                   |
| f8      | Permutation Entropy              |

In `nonlinear_analysis.py`, all 17 features have been implemented, including:

- Bandpower across δ/θ/α/β/γ
- Wavelet entropy
- Kurtosis
- Power signal
- Recurrence Quantification Analysis (RQA)
- ...

> **Note:** These legacy features are currently commented out but retained for future extension and reproducibility.

## 📤 Output Format

For each subject (`S*.mat`), the pipeline generates under OUTPUT_DIR:

- `{subject}_features.json`
- `{subject}_NL_Results.mat`
- `Creativity_NL_Data.json` and `.mat` as global aggregation
