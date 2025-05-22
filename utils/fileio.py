from scipy.io import loadmat
import logging
from pathlib import Path

logger = logging.getLogger("fileio")

def load_subject_mat(mat_path: Path):
    """Load MATLAB .mat, v7.3 自动切换到 mat73"""
    try:
        return loadmat(mat_path)
    except NotImplementedError:
        logger.info(f"{mat_path.name} 是 v7.3，改用 mat73 读取")
        import mat73
        return mat73.loadmat(mat_path)
