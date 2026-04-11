"""Independent FOSTER baseline package for thesis-oriented experiments."""

from .foster_client import FOSTERClient
from .foster_config import FOSTERConfig, get_foster_defaults
from .foster_external_generator import ClassConditionalFeatureGenerator
from .foster_server import FOSTERServer

__all__ = [
    "ClassConditionalFeatureGenerator",
    "FOSTERClient",
    "FOSTERConfig",
    "FOSTERServer",
    "get_foster_defaults",
]
