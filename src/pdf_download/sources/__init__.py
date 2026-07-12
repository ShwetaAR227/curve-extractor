"""Pluggable PDF-source registry: import a source module, register once here."""
from .base import PdfSource, get_source, iter_sources, list_sources, register
from .csv_url import CsvUrlSource
from .direct_mfr import DirectMfrSource
from .mouser_api import MouserApiSource

register(CsvUrlSource())
register(DirectMfrSource())
register(MouserApiSource())

__all__ = [
    "PdfSource", "get_source", "iter_sources", "list_sources", "register",
    "CsvUrlSource", "DirectMfrSource", "MouserApiSource",
]
