# utils/tqdm_events.py
import os, sys, json, time, itertools
from typing import Any, Dict, Optional
from tqdm import tqdm as _tqdm


class TqdmEvents(_tqdm):
    """
    A tqdm subclass that emits structured JSON events to stdout only.
    Events:
      - tqdm_start : when the bar is first displayed (lazy-emitted)
      - tqdm_update: on every display refresh
      - tqdm_meta  : on metadata changes (e.g., description)
      - tqdm_close : when the bar is closed
    Bars are distinguished by a unique 'bar_id' and position.
    """

    _BAR_SEQ = itertools.count(1)
    _PREFIX = os.getenv("EEG_TQDM_PREFIX", "EEG_TQDM")
    # Toggle: 1 = JSON only, 0 = also print the visual bar (call parent display)
    _JSON_ONLY = os.getenv("EEG_TQDM_VISUAL", "0") != "1"

    def __init__(self, *a, **k):
        # Force tqdm to write to stdout (default is stderr)
        k.setdefault("file", sys.stdout)
        k.setdefault("dynamic_ncols", True)
        k.setdefault("leave", True)

        # Assign bar_id early to avoid race conditions in multi-proc
        self.bar_id = f"{os.getpid()}-{next(self._BAR_SEQ)}"
        self._start_emitted = False  # lazy emit in display()

        super().__init__(*a, **k)

    # -------- core hook --------
    def display(self, msg: Optional[str] = None, pos: Optional[int] = None):
        """
        Override parent display so even the first refresh during __init__
        does NOT print a visual bar. We emit JSON events instead.
        """
        if not self._start_emitted:
            # Emit start exactly once, and before any update
            self._emit_tqdm({
                "type": "tqdm_start",
                "bar_id": self.bar_id,
                "pid": os.getpid(),
                "pos": getattr(self, "pos", 0),
                "desc": self.desc,
                "total": int(self.total) if self.total is not None else None,
                "unit": self.unit,
                "t": time.time(),
            })
            self._start_emitted = True

        # Gather live fields from format_dict
        d = self.format_dict
        evt = {
            "type": "tqdm_update",
            "bar_id": self.bar_id,
            "n": int(self.n),
            "total": int(self.total) if self.total is not None else None,
            "pos": getattr(self, "pos", 0),
            "desc": self.desc,
            "rate": d.get("rate"),
            "elapsed": d.get("elapsed"),
            "remaining": d.get("remaining"),
            "percentage": d.get("percentage"),
            "postfix": d.get("postfix"),
            "unit": self.unit,
            "t": time.time(),
        }
        self._emit_tqdm(evt)

        # If you ever want to keep visual output as well, allow via ENV VAR
        if not self._JSON_ONLY:
            # call parent to render the visual bar too
            super().display(msg=msg, pos=pos)

    # -------- small helpers --------
    def _emit_tqdm(self, evt: Dict[str, Any]):
        """Write one JSON line with prefix to stdout."""
        sys.stdout.write(self._PREFIX + " " + json.dumps(evt, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # -------- metadata hooks --------
    def set_description(self, desc=None, refresh=True):
        # If you don't want an extra tqdm_update caused by refresh, set refresh=False
        super().set_description(desc, refresh)
        self._emit_tqdm({
            "type": "tqdm_meta",
            "bar_id": self.bar_id,
            "desc": self.desc,
            "pos": getattr(self, "pos", 0),
            "t": time.time(),
        })

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        super().set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)
        self._emit_tqdm({
            "type": "tqdm_meta",
            "bar_id": self.bar_id,
            "postfix": self.format_dict.get("postfix"),
            "pos": getattr(self, "pos", 0),
            "t": time.time(),
        })

    def close(self):
        try:
            self._emit_tqdm({
                "type": "tqdm_close",
                "bar_id": self.bar_id,
                "n": int(self.n),
                "total": int(self.total) if self.total is not None else None,
                "pos": getattr(self, "pos", 0),
                "t": time.time(),
            })
        finally:
            super().close()
