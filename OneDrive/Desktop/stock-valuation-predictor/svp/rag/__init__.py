"""
Retrieval-augmented question answering over SEC filings.

  ``filings``  fetch 10-K / 10-Q text from EDGAR and split it into Items
  ``store``    chunk, embed and retrieve, with pluggable backends
  ``engine``   answer questions with quoted, cited bullets

Answers are extractive by default so every bullet is checkable against EDGAR;
an LLM synthesis layer is added only when an API key is configured.
"""

from . import filings, store, engine  # noqa: F401

__all__ = ["filings", "store", "engine"]
