import numpy as np
from logger import get_logger

logger = get_logger("nonlinear")

# These dependencies are optional; warn if missing
try:
    import antropy as ant
except ImportError:  # pragma: no cover
    ant = None
    logger.warning("antropy not installed – entropy measures unavailable")

try:
    import nolds
except ImportError:  # pragma: no cover
    nolds = None
    logger.warning("nolds not installed – fractal/chaos measures unavailable")

# Map to the correct function name depending on AntroPy version
if ant is not None:
    if hasattr(ant, "permutation_entropy"):
        _perm_entropy_func = ant.permutation_entropy
    elif hasattr(ant, "perm_entropy"):            # old API
        _perm_entropy_func = ant.perm_entropy
    else:
        _perm_entropy_func = None
        logger.warning("AntroPy has no permutation entropy implementation.")
else:
    _perm_entropy_func = None

def _choose_higuchi():
    if nolds is not None and hasattr(nolds, "higuchi_fd"):
        return nolds.higuchi_fd
    if ant is not None and hasattr(ant, "higuchi_fd"):
        return ant.higuchi_fd
    return None

_HIGUCHI_F = _choose_higuchi()

def sample_entropy(sig):
    if ant is None:
        return np.nan
    return ant.sample_entropy(sig)


def permutation_entropy(sig, order=3, delay=1):
    """Return normalized permutation entropy or NaN if unavailable."""
    if _perm_entropy_func is None:
        return np.nan
    return _perm_entropy_func(sig, order=order, delay=delay, normalize=True)


def higuchi_fd(sig, kmax=8):
    """Return Higuchi fractal dimension or NaN if unavailable."""
    if _HIGUCHI_F is None:
        logger.warning("No Higuchi FD implementation found; returning NaN.")
        return np.nan
    return _HIGUCHI_F(sig, kmax=kmax)


def lyapunov_exponent(sig):
    if nolds is None:
        return np.nan
    try:
        return nolds.lyap_e(sig)[0]
    except Exception as exc:
        logger.warning(f"Lyapunov computation failed: {exc}")
        return np.nan
