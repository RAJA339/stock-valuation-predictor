"""Configuration resolution and naming rules. No Spark required."""

from __future__ import annotations

import json

import pytest

from pipelines.common.config import ConfigError, PipelineConfig


def make(**overrides) -> PipelineConfig:
    base = {
        "environment": "dev",
        "catalog": "genai_lakehouse_dev",
        "checkpoint_path": "/Volumes/genai_lakehouse_dev/bronze/checkpoints",
    }
    base.update(overrides)
    return PipelineConfig(**base)


class TestNaming:
    def test_table_uses_three_level_namespace(self):
        config = make()
        assert config.bronze("raw_documents") == "genai_lakehouse_dev.bronze.raw_documents"
        assert config.silver("market_events") == "genai_lakehouse_dev.silver.market_events"
        assert config.gold("document_context") == "genai_lakehouse_dev.gold.document_context"

    def test_unknown_layer_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown medallion layer"):
            make().table("platinum", "nope")

    def test_checkpoint_paths_are_per_stream(self):
        config = make()
        a = config.checkpoint("bronze_documents")
        b = config.checkpoint("bronze_market_events")
        assert a != b, "two streams sharing a checkpoint corrupt each other's offsets"
        assert a.endswith("/dev/bronze_documents")

    def test_checkpoint_requires_a_stream_name(self):
        with pytest.raises(ConfigError):
            make().checkpoint("")

    def test_checkpoint_requires_configured_base_path(self):
        with pytest.raises(ConfigError, match="checkpoint_path"):
            make(checkpoint_path="").checkpoint("some_stream")

    def test_index_name_is_derived_from_source_table(self):
        config = make()
        source = config.gold("document_context")
        assert config.index_name(source) == f"{source}_vs_index"


class TestValidation:
    def test_environment_must_be_known(self):
        with pytest.raises(ConfigError, match="environment must be one of"):
            make(environment="qa")

    def test_catalog_is_required(self):
        with pytest.raises(ConfigError, match="catalog must be set"):
            make(catalog="")

    @pytest.mark.parametrize("size,overlap", [(100, 100), (100, 250)])
    def test_overlap_must_be_smaller_than_chunk_size(self, size, overlap):
        # An overlap >= chunk_size means the chunker never advances.
        with pytest.raises(ConfigError, match="chunk_overlap"):
            make(chunk_size=size, chunk_overlap=overlap)


class TestConstruction:
    def test_from_dict_coerces_terraform_strings(self):
        config = PipelineConfig.from_dict(
            {
                "environment": "prod",
                "catalog": "c",
                "chunk_size": "800",
                "chunk_overlap": "100",
                "embedding_dimension": "1024",
            }
        )
        assert config.chunk_size == 800
        assert config.embedding_dimension == 1024

    def test_unknown_keys_land_in_extra(self):
        config = PipelineConfig.from_dict(
            {"environment": "dev", "catalog": "c", "use_file_notifications": "true"}
        )
        assert config.extra["use_file_notifications"] == "true"

    def test_from_json_file_accepts_raw_block(self, tmp_path):
        path = tmp_path / "conf.json"
        path.write_text(json.dumps({"environment": "dev", "catalog": "raw_block"}))
        assert PipelineConfig.from_json_file(path).catalog == "raw_block"

    def test_from_json_file_accepts_terraform_output_dump(self, tmp_path):
        # `terraform output -json` wraps every output in {"value": ..., "type": ...}
        path = tmp_path / "tf.json"
        path.write_text(
            json.dumps(
                {
                    "pipeline_config": {
                        "value": {"environment": "prod", "catalog": "from_tf"},
                        "type": "object",
                    }
                }
            )
        )
        assert PipelineConfig.from_json_file(path).catalog == "from_tf"

    def test_from_env_reads_prefixed_variables(self):
        config = PipelineConfig.from_env(
            {"LAKEHOUSE_ENVIRONMENT": "dev", "LAKEHOUSE_CATALOG": "c", "OTHER": "ignored"}
        )
        assert config.catalog == "c"
        assert "other" not in config.extra

    def test_from_env_without_variables_is_an_error(self):
        with pytest.raises(ConfigError, match="no LAKEHOUSE_"):
            PipelineConfig.from_env({})

    def test_resolve_prefers_conf_argument(self, tmp_path, monkeypatch):
        path = tmp_path / "conf.json"
        path.write_text(json.dumps({"environment": "dev", "catalog": "from_arg"}))
        monkeypatch.setenv("LAKEHOUSE_CATALOG", "from_env")
        monkeypatch.setenv("LAKEHOUSE_ENVIRONMENT", "dev")
        assert PipelineConfig.resolve(["--conf", str(path)]).catalog == "from_arg"

    def test_config_is_immutable(self):
        # A frozen dataclass raises FrozenInstanceError, which subclasses
        # AttributeError. Config mutated mid-run would mean two streams in the
        # same driver writing to different catalogs.
        config = make()
        with pytest.raises(AttributeError):
            config.catalog = "something_else"  # type: ignore[misc]
