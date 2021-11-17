"""Collection of output format classes. Separated so that loading causes no import errors in any environment."""
from dataclasses import dataclass
from typing import Any, MutableMapping


@dataclass
class SRLOutput:
    """Class to hold SRL Output."""

    label_id: int
    verb: str
    args: MutableMapping[str, str]


@dataclass
class AMROutput:
    """Class to hold AMR Output."""

    label_id: int
    graph: Any
    alignments: Any
    annotations: Any
