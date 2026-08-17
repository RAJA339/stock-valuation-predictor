"""Runtime configuration for every pipeline in the platform.

Configuration has exactly one shape (:class:`PipelineConfig`) and three
possible sources, resolved in this order:

1. an explicit ``--conf path/to/config.json`` argument,
2. the ``LAKEHOUSE_CONFIG`` environment variable pointing at the same JSON,
3. individual ``LAKEHOUSE_*`` environment variables.

The JSON file is the ``pipeline_config`` Terraform output, written verbatim.
Keeping the contract that narrow means a job can never drift from the
infrastructure that was actually deployed -- if Terraform renames a catalog,
the next deploy rewrites the config and the job picks it up.

Nothing in this module imports pyspark, so it is cheap to unit test.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ENV_PREFIX = "LAKEHOUSE_"

VALID_ENVIRONMENTS = ("dev", "staging", "prod")


class ConfigError(ValueError):
    """Raised when the resolved configuration is unusable."""


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a pipeline needs to know about where it reads and writes.

    Paths are Unity Catalog volume paths rather than raw ``s3://`` URIs on
    purpose: a job that addresses S3 directly bypasses UC permissions and
    lineage, which is precisely what the governance layer exists to prevent.
    """

    environment: str
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"

    # Ingest
    landing_path: str = ""
    checkpoint_path: str = ""
    kinesis_stream: str = ""
    kinesis_region: str = "us-east-1"
    kinesis_initial_position: str = "TRIM_HORIZON"

    # Vector search
    vector_endpoint: str = ""
    embedding_model_endpoint: str = "databricks-gte-large-en"
    embedding_dimension: int = 1024

    # Streaming knobs
    max_files_per_trigger: int = 1000
    max_bytes_per_trigger: str = "1g"
    trigger_interval: str = "30 seconds"

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200

    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- validation --------------------------------------------------------

    def __post_init__(self) -> None:
        if self.environment not in VALID_ENVIRONMENTS:
            raise ConfigError(
                f"environment must be one of {VALID_ENVIRONMENTS}, got {self.environment!r}"
            )
        if not self.catalog:
            raise ConfigError("catalog must be set")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunking never advances"
            )

    # -- naming helpers ----------------------------------------------------

    def table(self, layer: str, name: str) -> str:
        """Return the three-level namespace name for a table.

        >>> PipelineConfig(environment="dev", catalog="c").table("bronze", "raw_documents")
        'c.bronze.raw_documents'
        """
        schema = {
            "bronze": self.bronze_schema,
            "silver": self.silver_schema,
            "gold": self.gold_schema,
        }.get(layer)
        if schema is None:
            raise ConfigError(f"unknown medallion layer {layer!r}")
        return f"{self.catalog}.{schema}.{name}"

    def bronze(self, name: str) -> str:
        return self.table("bronze", name)

    def silver(self, name: str) -> str:
        return self.table("silver", name)

    def gold(self, name: str) -> str:
        return self.table("gold", name)

    def checkpoint(self, stream_name: str) -> str:
        """Checkpoint location for a named stream.

        Two streams sharing a checkpoint corrupt each other's offsets, so the
        stream name is always part of the path -- never let a caller pass a
        bare directory.
        """
        if not stream_name:
            raise ConfigError("stream_name is required to build a checkpoint path")
        base = self.checkpoint_path.rstrip("/")
        if not base:
            raise ConfigError("checkpoint_path is not configured")
        return f"{base}/{self.environment}/{stream_name}"

    def index_name(self, source_table: str) -> str:
        """Vector Search index name derived from its source table."""
        return f"{source_table}_vs_index"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PipelineConfig:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        # Terraform renders every output as a string; coerce the numeric ones
        # rather than blowing up deep inside a streaming query.
        for int_field in (
            "embedding_dimension",
            "max_files_per_trigger",
            "chunk_size",
            "chunk_overlap",
        ):
            if int_field in kwargs and kwargs[int_field] is not None:
                kwargs[int_field] = int(kwargs[int_field])
        return cls(extra=extra, **kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> PipelineConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # Accept both the raw block and a full `terraform output -json` dump.
        if "pipeline_config" in payload and isinstance(payload["pipeline_config"], dict):
            block = payload["pipeline_config"]
            payload = block.get("value", block)
        return cls.from_dict(payload)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PipelineConfig:
        env = os.environ if env is None else env
        raw = {
            key[len(_ENV_PREFIX) :].lower(): value
            for key, value in env.items()
            if key.startswith(_ENV_PREFIX) and key != f"{_ENV_PREFIX}CONFIG"
        }
        if not raw:
            raise ConfigError(
                f"no {_ENV_PREFIX}* variables found; set {_ENV_PREFIX}CONFIG or pass --conf"
            )
        return cls.from_dict(raw)

    @classmethod
    def resolve(cls, argv: list[str] | None = None) -> PipelineConfig:
        """Resolve configuration from CLI args, then env, then env vars."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--conf", dest="conf", default=None)
        args, _unknown = parser.parse_known_args(argv)

        path = args.conf or os.environ.get(f"{_ENV_PREFIX}CONFIG")
        if path:
            return cls.from_json_file(path)
        return cls.from_env()
