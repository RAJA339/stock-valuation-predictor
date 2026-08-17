"""Layer transforms: Bronze parsing, Silver conforming, Gold aggregation.

Streaming sources and Delta writes are out of scope here -- those need a
cluster. What is tested is the pure DataFrame -> DataFrame function each
pipeline is built around, which is where the business logic actually lives.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from pyspark.sql import functions as F

from pipelines.bronze import autoloader_documents as autoloader
from pipelines.bronze import kinesis_market_events as kinesis
from pipelines.gold import gold_aggregates as gold
from pipelines.silver import market_events_silver as silver_events
from pipelines.silver.instrument_scd2 import transform as instrument_transform

# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------


class TestKinesisTransform:
    @pytest.fixture
    def kinesis_batch(self, spark, now):
        def record(payload: str, key: str = "AAPL"):
            return (bytearray(payload.encode("utf-8")), key, "seq-1", now)

        good = json.dumps(
            {
                "event_id": "e1",
                "event_type": "trade",
                "symbol": "AAPL",
                "occurred_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "price": 190.5,
                "volume": 100,
                "bid": 190.4,
                "ask": 190.6,
                "venue": "XNAS",
                "source_system": "feed-a",
            }
        )
        return spark.createDataFrame(
            [record(good), record("{not valid json", "BAD")],
            "data BINARY, partitionKey STRING, sequenceNumber STRING, "
            "approximateArrivalTimestamp TIMESTAMP",
        )

    def test_parses_a_well_formed_payload(self, kinesis_batch):
        out = kinesis.transform(kinesis_batch, run_id="r")
        row = out.filter(F.col("event_id") == "e1").first()
        assert row.symbol == "AAPL"
        assert row.price == pytest.approx(190.5)
        assert row.occurred_at is not None
        assert row._corrupt_record is None

    def test_malformed_payload_is_flagged_not_dropped(self, kinesis_batch):
        out = kinesis.transform(kinesis_batch, run_id="r")
        corrupt = out.filter(F.col("_corrupt_record").isNotNull()).collect()
        assert len(corrupt) == 1
        assert corrupt[0].event_id is None

    def test_raw_payload_is_always_preserved(self, kinesis_batch):
        # When a producer breaks its contract, the raw bytes are the only way
        # to replay the data after the parser is fixed.
        out = kinesis.transform(kinesis_batch, run_id="r")
        assert out.filter(F.col("_raw_payload").isNull()).count() == 0

    def test_kinesis_metadata_is_carried_through(self, kinesis_batch):
        out = kinesis.transform(kinesis_batch, run_id="r")
        row = out.first()
        assert row._kinesis_sequence_number == "seq-1"
        assert row._kinesis_partition_key in ("AAPL", "BAD")
        assert row._pipeline_name == kinesis.PIPELINE_NAME


class TestAutoloaderTransform:
    @pytest.fixture
    def files(self, spark, now):
        return spark.createDataFrame(
            [
                ("s3://bucket/landing/a/report.txt", now, 42, bytearray(b"annual report text")),
                ("s3://bucket/landing/b/deck.pdf", now, 99, bytearray(b"%PDF-1.7 binary")),
            ],
            "path STRING, modificationTime TIMESTAMP, length LONG, content BINARY",
        )

    def test_decodes_text_formats(self, files):
        out = autoloader.transform(files, run_id="r")
        row = out.filter(F.col("file_extension") == "txt").first()
        assert row.content == "annual report text"
        assert row.file_name == "report.txt"

    def test_leaves_binary_formats_undecoded(self, files):
        # A PDF decoded as UTF-8 is mojibake; parsing happens in Silver.
        out = autoloader.transform(files, run_id="r")
        assert out.filter(F.col("file_extension") == "pdf").first().content is None

    def test_document_id_is_the_content_hash(self, spark, files, now):
        out = autoloader.transform(files, run_id="r")
        for row in out.collect():
            assert row.document_id == row.content_sha256

        # Same bytes under a different path must yield the same document_id --
        # that is what makes re-uploads idempotent all the way to the index.
        renamed = spark.createDataFrame(
            [("s3://bucket/landing/c/copy.txt", now, 42, bytearray(b"annual report text"))],
            "path STRING, modificationTime TIMESTAMP, length LONG, content BINARY",
        )
        original_id = out.filter(F.col("file_extension") == "txt").first().document_id
        assert autoloader.transform(renamed, run_id="r").first().document_id == original_id

    def test_output_matches_the_declared_schema(self, files):
        from pipelines.common.schemas import BRONZE_DOCUMENTS_SCHEMA

        out = autoloader.transform(files, run_id="r")
        assert out.columns == [f.name for f in BRONZE_DOCUMENTS_SCHEMA.fields]


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------


class TestSilverMarketEvents:
    def test_deduplicates_on_event_id(self, market_events):
        out = silver_events.transform(market_events, run_id="r")
        assert out.filter(F.col("event_id") == "e1").count() == 1

    def test_duplicate_resolution_keeps_the_latest_event(self, market_events):
        out = silver_events.transform(market_events, run_id="r")
        assert out.filter(F.col("event_id") == "e1").first().price == pytest.approx(191.0)

    def test_normalises_symbol_and_venue(self, market_events):
        out = silver_events.transform(market_events, run_id="r")
        symbols = {r.symbol for r in out.collect() if r.symbol}
        assert symbols <= {"AAPL", "MSFT", "TSLA", "NVDA"}
        assert all(r.venue is None or r.venue.isupper() for r in out.collect())

    def test_derives_event_date_from_occurred_at(self, market_events, now):
        out = silver_events.transform(market_events, run_id="r")
        assert out.first().event_date == now.strftime("%Y-%m-%d")

    def test_computes_spread(self, market_events):
        out = silver_events.transform(market_events, run_id="r")
        row = out.filter(F.col("event_id") == "e2").first()
        assert row.spread == pytest.approx(0.3, abs=1e-6)

    def test_output_matches_the_declared_schema(self, market_events):
        from pipelines.common.schemas import SILVER_MARKET_EVENTS_SCHEMA

        out = silver_events.transform(market_events, run_id="r")
        assert out.columns == [f.name for f in SILVER_MARKET_EVENTS_SCHEMA.fields]


class TestInstrumentTransform:
    def test_conforms_casing_and_types(self, spark):
        raw = spark.createDataFrame(
            [(None, " aapl ", "Apple Inc", "technology", "us", "usd", "XNAS", "true")],
            "instrument_key STRING, symbol STRING, issuer_name STRING, sector STRING, "
            "country STRING, currency STRING, listing_venue STRING, is_active STRING",
        )
        row = instrument_transform(raw, run_id="r").first()
        assert row.symbol == "AAPL"
        assert row.currency == "USD"
        assert row.country == "US"
        assert row.sector == "Technology"
        assert row.is_active is True

    def test_instrument_key_falls_back_to_symbol(self, spark):
        raw = spark.createDataFrame(
            [(None, "MSFT", "Microsoft", "Technology", "US", "USD", "XNAS", True)],
            "instrument_key STRING, symbol STRING, issuer_name STRING, sector STRING, "
            "country STRING, currency STRING, listing_venue STRING, is_active BOOLEAN",
        )
        assert instrument_transform(raw, run_id="r").first().instrument_key == "MSFT"


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------


@pytest.fixture
def silver_events_df(spark, now):
    rows = [
        ("t1", "trade", "AAPL", now, "2026-08-17", 100.0, 10, 0.1),
        ("t2", "trade", "AAPL", now + timedelta(minutes=1), "2026-08-17", 110.0, 20, 0.2),
        ("t3", "trade", "AAPL", now + timedelta(minutes=2), "2026-08-17", 90.0, 30, 0.3),
        ("t4", "trade", "MSFT", now, "2026-08-17", 400.0, 5, 0.4),
    ]
    return spark.createDataFrame(
        rows,
        "event_id STRING, event_type STRING, symbol STRING, occurred_at TIMESTAMP, "
        "event_date STRING, price DOUBLE, volume LONG, spread DOUBLE",
    )


@pytest.fixture
def instrument_dim(spark, now):
    """AAPL was reclassified: Consumer until an hour ago, Technology since."""
    reclass = now - timedelta(minutes=30)
    rows = [
        ("AAPL", "Consumer", now - timedelta(days=30), reclass, False),
        ("AAPL", "Technology", reclass, datetime(9999, 12, 31, 23, 59, 59), True),
        ("MSFT", "Technology", now - timedelta(days=30), datetime(9999, 12, 31, 23, 59, 59), True),
    ]
    return spark.createDataFrame(
        rows,
        "symbol STRING, sector STRING, valid_from TIMESTAMP, valid_to TIMESTAMP, is_current BOOLEAN",
    )


class TestAsOfJoin:
    def test_matches_exactly_one_version(self, silver_events_df, instrument_dim):
        joined = gold.as_of_join(
            silver_events_df,
            instrument_dim,
            fact_key="symbol",
            dim_key="symbol",
            fact_time="occurred_at",
        )
        assert joined.count() == silver_events_df.count(), "an as-of join must not fan out"

    def test_uses_the_version_current_at_event_time(
        self, spark, silver_events_df, instrument_dim, now
    ):
        # An event from before the reclassification must still read "Consumer".
        historical = silver_events_df.filter(F.col("event_id") == "t1").withColumn(
            "occurred_at", F.lit(now - timedelta(hours=2)).cast("timestamp")
        )
        joined = gold.as_of_join(
            historical, instrument_dim, fact_key="symbol", dim_key="symbol", fact_time="occurred_at"
        )
        assert joined.select("d.sector").first().sector == "Consumer"

    def test_recent_events_read_the_current_version(self, silver_events_df, instrument_dim):
        joined = gold.as_of_join(
            silver_events_df.filter(F.col("event_id") == "t1"),
            instrument_dim,
            fact_key="symbol",
            dim_key="symbol",
            fact_time="occurred_at",
        )
        assert joined.select("d.sector").first().sector == "Technology"


class TestSymbolDailyMetrics:
    def test_aggregates_per_symbol_and_date(self, silver_events_df, instrument_dim):
        out = gold.build_symbol_daily_metrics(silver_events_df, instrument_dim)
        assert out.count() == 2

    def test_ohlc_reflects_event_order(self, silver_events_df, instrument_dim):
        row = (
            gold.build_symbol_daily_metrics(silver_events_df, instrument_dim)
            .filter(F.col("symbol") == "AAPL")
            .first()
        )
        assert row.open_price == pytest.approx(100.0)
        assert row.close_price == pytest.approx(90.0)
        assert row.high_price == pytest.approx(110.0)
        assert row.low_price == pytest.approx(90.0)

    def test_vwap_is_volume_weighted_not_a_mean_of_prices(self, silver_events_df, instrument_dim):
        row = (
            gold.build_symbol_daily_metrics(silver_events_df, instrument_dim)
            .filter(F.col("symbol") == "AAPL")
            .first()
        )
        expected = (100.0 * 10 + 110.0 * 20 + 90.0 * 30) / 60
        assert row.vwap == pytest.approx(expected)
        assert row.vwap != pytest.approx((100.0 + 110.0 + 90.0) / 3)

    def test_volume_is_summed(self, silver_events_df, instrument_dim):
        row = (
            gold.build_symbol_daily_metrics(silver_events_df, instrument_dim)
            .filter(F.col("symbol") == "AAPL")
            .first()
        )
        assert row.total_volume == 60
        assert row.event_count == 3

    def test_output_matches_the_declared_schema(self, silver_events_df, instrument_dim):
        from pipelines.common.schemas import GOLD_SYMBOL_DAILY_METRICS_SCHEMA

        out = gold.build_symbol_daily_metrics(silver_events_df, instrument_dim)
        assert set(out.columns) == {f.name for f in GOLD_SYMBOL_DAILY_METRICS_SCHEMA.fields}


class TestDocumentContext:
    @pytest.fixture
    def chunks(self, spark, now):
        return spark.createDataFrame(
            [
                (
                    "c1",
                    "d1",
                    0,
                    "Strong results at $AAPL this quarter.",
                    "s3://x/2026-03-01/a.txt",
                    now,
                ),
                (
                    "c2",
                    "d1",
                    1,
                    "No ticker mentioned in this passage.",
                    "s3://x/2026-03-01/a.txt",
                    now,
                ),
            ],
            "chunk_id STRING, document_id STRING, chunk_index INT, chunk_text STRING, "
            "source_path STRING, _processed_at TIMESTAMP",
        )

    def test_extracts_the_mentioned_symbol(self, chunks, instrument_dim):
        out = gold.build_document_context(chunks, instrument_dim)
        assert out.filter(F.col("chunk_id") == "c1").first().symbol == "AAPL"

    def test_chunk_without_a_ticker_keeps_a_null_symbol(self, chunks, instrument_dim):
        out = gold.build_document_context(chunks, instrument_dim)
        assert out.filter(F.col("chunk_id") == "c2").first().symbol is None

    def test_joins_the_current_sector(self, chunks, instrument_dim):
        out = gold.build_document_context(chunks, instrument_dim)
        assert out.filter(F.col("chunk_id") == "c1").first().sector == "Technology"

    def test_published_date_is_taken_from_the_path(self, chunks, instrument_dim):
        out = gold.build_document_context(chunks, instrument_dim)
        assert out.first().published_date == "2026-03-01"

    def test_does_not_fan_out_rows(self, chunks, instrument_dim):
        out = gold.build_document_context(chunks, instrument_dim)
        assert out.count() == chunks.count()

    def test_carries_every_column_the_index_syncs(self, chunks, instrument_dim):
        from pipelines.gold.vector_index import SYNCED_COLUMNS

        out = gold.build_document_context(chunks, instrument_dim)
        missing = set(SYNCED_COLUMNS) - set(out.columns)
        assert not missing, f"vector index would fail to sync: {missing}"
