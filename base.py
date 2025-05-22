from .logger import get_logger


class PipelineBase:
    """Base class giving every pipeline stage a logger."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
