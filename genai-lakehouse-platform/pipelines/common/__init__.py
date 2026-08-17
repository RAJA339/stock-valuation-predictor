"""Shared building blocks for every medallion pipeline.

Import from here rather than from the submodules so the public surface of the
package stays small and refactorable.
"""

from .config import ConfigError, PipelineConfig
from .quality import (
    Action,
    DataQualityError,
    Expectation,
    QualityMetrics,
    ValidationResult,
    validate,
)
from .scd2 import Scd2Error, Scd2Spec, apply_scd2_merge, build_staged_updates, prepare_source
from .text import Chunk, chunk_text, clean_text, content_hash, estimate_tokens

__all__ = [
    "Action",
    "Chunk",
    "ConfigError",
    "DataQualityError",
    "Expectation",
    "PipelineConfig",
    "QualityMetrics",
    "Scd2Error",
    "Scd2Spec",
    "ValidationResult",
    "apply_scd2_merge",
    "build_staged_updates",
    "chunk_text",
    "clean_text",
    "content_hash",
    "estimate_tokens",
    "prepare_source",
    "validate",
]
