"""Shared DataFrame transforms."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import functions as F

from pipelines.common.transforms import (
    add_spread_metrics,
    chunk_documents,
    deduplicate,
    new_run_id,
    normalise_symbol,
    parse_timestamp,
    with_ingest_metadata,
    with_processing_metadata,
)


class TestRunId:
    def test_ids_are_unique(self):
        assert len({new_run_id() for _ in range(100)}) == 100


class TestMetadataStamping:
    def test_ingest_metadata_adds_every_audit_column(self, spark):
        df = spark.createDataFrame([("x",)], "path STRING")
        out = with_ingest_metadata(df, pipeline_name="p", run_id="r1", source_file_col="path")
        row = out.first()
        assert row._pipeline_name == "p"
        assert row._pipeline_run_id == "r1"
        assert row._source_file == "x"
        assert row._ingest_date == row._ingest_timestamp.strftime("%Y-%m-%d")

    def test_source_file_is_null_when_not_applicable(self, spark):
        df = spark.createDataFrame([(1,)], "id INT")
        assert with_ingest_metadata(df, pipeline_name="p", run_id="r").first()._source_file is None

    def test_processing_metadata_stamps_the_run(self, spark):
        df = spark.createDataFrame([(1,)], "id INT")
        assert with_processing_metadata(df, run_id="r2").first()._pipeline_run_id == "r2"


class TestDeduplicate:
    def test_keeps_the_winner_by_order(self, spark):
        df = spark.createDataFrame(
            [("k", 1), ("k", 3), ("k", 2), ("j", 9)], "key STRING, version INT"
        )
        out = deduplicate(df, ["key"], F.col("version").desc())
        assert {r.key: r.version for r in out.collect()} == {"k": 3, "j": 9}

    def test_supports_multiple_order_columns(self, spark):
        df = spark.createDataFrame([("k", 1, "b"), ("k", 1, "a")], "key STRING, v INT, tie STRING")
        out = deduplicate(df, ["key"], [F.col("v").desc(), F.col("tie").asc()])
        assert out.first().tie == "a"

    def test_composite_keys(self, spark):
        df = spark.createDataFrame(
            [("a", "x", 1), ("a", "x", 2), ("a", "y", 1)], "k1 STRING, k2 STRING, v INT"
        )
        assert deduplicate(df, ["k1", "k2"], F.col("v").desc()).count() == 2

    def test_requires_a_key(self, spark):
        df = spark.createDataFrame([(1,)], "id INT")
        with pytest.raises(ValueError):
            deduplicate(df, [], F.col("id").asc())


class TestNormaliseSymbol:
    @pytest.mark.parametrize(
        "raw,expected",
        [("aapl", "AAPL"), ("  msft ", "MSFT"), ("BrK.b", "BRK.B"), (None, None)],
    )
    def test_normalises(self, spark, raw, expected):
        df = spark.createDataFrame([(raw,)], "symbol STRING")
        assert df.select(normalise_symbol(F.col("symbol")).alias("s")).first().s == expected

    def test_blank_becomes_null(self, spark):
        # '' and NULL mean the same thing; having both makes every GROUP BY
        # and join subtly wrong.
        df = spark.createDataFrame([("   ",)], "symbol STRING")
        assert df.select(normalise_symbol(F.col("symbol")).alias("s")).first().s is None


class TestParseTimestamp:
    def test_parses_iso8601(self, spark):
        df = spark.createDataFrame([("2026-08-17T10:30:00",)], "ts STRING")
        assert df.select(parse_timestamp(F.col("ts")).alias("t")).first().t == datetime(
            2026, 8, 17, 10, 30
        )

    def test_parses_epoch_millis(self, spark):
        df = spark.createDataFrame([("1755426600000",)], "ts STRING")
        parsed = df.select(parse_timestamp(F.col("ts")).alias("t")).first().t
        assert parsed is not None and parsed.year == 2025

    def test_garbage_becomes_null_instead_of_failing(self, spark):
        # A stream must not die because one producer sent nonsense; the
        # quality gate reports it instead.
        df = spark.createDataFrame([("not a timestamp",)], "ts STRING")
        assert df.select(parse_timestamp(F.col("ts")).alias("t")).first().t is None


class TestSpreadMetrics:
    def test_computes_spread_and_mid(self, spark):
        df = spark.createDataFrame([(10.0, 10.4)], "bid DOUBLE, ask DOUBLE")
        row = add_spread_metrics(df).first()
        assert row.spread == pytest.approx(0.4)
        assert row.mid_price == pytest.approx(10.2)

    def test_zero_mid_is_null_not_zero(self, spark):
        df = spark.createDataFrame([(0.0, 0.0)], "bid DOUBLE, ask DOUBLE")
        assert add_spread_metrics(df).first().mid_price is None

    def test_missing_side_yields_null(self, spark):
        df = spark.createDataFrame([(None, 10.0)], "bid DOUBLE, ask DOUBLE")
        row = add_spread_metrics(df).first()
        assert row.spread is None and row.mid_price is None


class TestChunkDocuments:
    @pytest.fixture
    def documents(self, spark):
        long_text = "Revenue grew materially across every reporting segment. " * 40
        return spark.createDataFrame(
            [("doc1", long_text), ("doc2", "A single short paragraph about guidance.")],
            "document_id STRING, content STRING",
        )

    def test_explodes_into_multiple_rows(self, documents):
        out = chunk_documents(documents, chunk_size=400, chunk_overlap=50)
        assert out.count() > 2
        assert out.filter(F.col("document_id") == "doc1").count() > 1

    def test_chunk_ids_are_unique_and_deterministic(self, documents):
        first = chunk_documents(documents, chunk_size=400, chunk_overlap=50)
        second = chunk_documents(documents, chunk_size=400, chunk_overlap=50)
        ids_a = sorted(r.chunk_id for r in first.collect())
        ids_b = sorted(r.chunk_id for r in second.collect())
        assert ids_a == ids_b, "a re-run must not create duplicate index entries"
        assert len(ids_a) == len(set(ids_a))

    def test_chunk_index_restarts_per_document(self, documents):
        out = chunk_documents(documents, chunk_size=400, chunk_overlap=50)
        for doc in ("doc1", "doc2"):
            indices = sorted(
                r.chunk_index for r in out.filter(F.col("document_id") == doc).collect()
            )
            assert indices == list(range(len(indices)))

    def test_carries_source_columns_through(self, spark):
        df = spark.createDataFrame(
            [
                (
                    "d",
                    "Some reasonably long content that will survive the minimum length gate.",
                    "s3://x",
                )
            ],
            "document_id STRING, content STRING, path STRING",
        )
        assert "path" in chunk_documents(df).columns

    def test_document_with_no_usable_text_drops_out(self, spark):
        df = spark.createDataFrame([("d", "")], "document_id STRING, content STRING")
        assert chunk_documents(df).count() == 0
