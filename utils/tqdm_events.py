# utils/tqdm_events.py
import os, sys, json, time, itertools
from typing import Any, Dict
from tqdm import tqdm as _tqdm


class TqdmEvents(_tqdm):
    """
    A tqdm subclass that emits structured JSON events to stdout.
    Events:
      - tqdm_start : when the bar is created
      - tqdm_update: on every display refresh
      - tqdm_meta  : on metadata changes (e.g. description)
      - tqdm_close : when the bar is closed
    Bars are distinguished by a unique 'bar_id' and position.
    """

    _BAR_SEQ = itertools.count(1)
    _PREFIX = os.getenv("EEG_TQDM_PREFIX", "EEG_TQDM")
    #_VISUAL = os.getenv("EEG_TQDM_VISUAL", "0") == "1"  # 1=show bar, 0=JSON only

    def __init__(self, *a, **k):
        # Force tqdm to write to stdout (default is stderr)
        k.setdefault("file", sys.stdout)
        k.setdefault("dynamic_ncols", True)
        k.setdefault("leave", True)

        # Assign bar_id early to avoid race conditions
        self.bar_id = f"{os.getpid()}-{next(self._BAR_SEQ)}"

        # Let the parent class initialize its internal state
        super().__init__(*a, **k)

        # Emit start event
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

        # Wrap display() AFTER init is finished
        _parent_display = super(TqdmEvents, self).display

        def _hooked_display(msg=None, pos=None):
            # if self._VISUAL:
            #     _parent_display(msg, pos)  # keep the normal progress bar
            # Always emit JSON event
            d = self.format_dict
            self._emit_tqdm({
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
                "unit": self.unit,
                "t": time.time(),
            })

        self.display = _hooked_display

    def _emit_tqdm(self, evt: Dict[str, Any]):
        """Write one JSON line with prefix to stdout."""
        sys.stdout.write(self._PREFIX + " " + json.dumps(evt, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def set_description(self, desc=None, refresh=True):
        super().set_description(desc, refresh)
        # Emit metadata change
        self._emit_tqdm({
            "type": "tqdm_meta",
            "bar_id": self.bar_id,
            "desc": self.desc,
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
