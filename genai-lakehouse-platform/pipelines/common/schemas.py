"""Explicit schemas for every table the platform writes.

Schema inference is banned outside the Bronze landing zone. Inference on a
streaming source means the schema is whatever the first micro-batch happened
to contain, which turns an upstream producer's bad day into a silent column
type change three layers downstream.
"""

from __future__ import annotations

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------

#: Payload carried on the Kinesis stream. Parsed with mode PERMISSIVE so a
#: malformed record lands in ``_corrupt_record`` and is quarantined rather than
#: killing the stream.
MARKET_EVENT_PAYLOAD_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), nullable=False),
        StructField("event_type", StringType(), nullable=False),
        StructField("symbol", StringType(), nullable=False),
        StructField("occurred_at", StringType(), nullable=False),
        StructField("price", DoubleType(), nullable=True),
        StructField("volume", LongType(), nullable=True),
        StructField("bid", DoubleType(), nullable=True),
        StructField("ask", DoubleType(), nullable=True),
        StructField("venue", StringType(), nullable=True),
        StructField("source_system", StringType(), nullable=True),
        StructField("attributes", MapType(StringType(), StringType()), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True),
    ]
)

#: Columns every Bronze table carries, regardless of source. Written by
#: :func:`pipelines.common.transforms.with_ingest_metadata`.
BRONZE_AUDIT_FIELDS = [
    StructField("_ingest_timestamp", TimestampType(), nullable=False),
    StructField("_ingest_date", StringType(), nullable=False),
    StructField("_source_file", StringType(), nullable=True),
    StructField("_pipeline_run_id", StringType(), nullable=False),
    StructField("_pipeline_name", StringType(), nullable=False),
]

BRONZE_MARKET_EVENTS_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), nullable=True),
        StructField("event_type", StringType(), nullable=True),
        StructField("symbol", StringType(), nullable=True),
        StructField("occurred_at", TimestampType(), nullable=True),
        StructField("price", DoubleType(), nullable=True),
        StructField("volume", LongType(), nullable=True),
        StructField("bid", DoubleType(), nullable=True),
        StructField("ask", DoubleType(), nullable=True),
        StructField("venue", StringType(), nullable=True),
        StructField("source_system", StringType(), nullable=True),
        StructField("attributes", MapType(StringType(), StringType()), nullable=True),
        StructField("_raw_payload", StringType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True),
        StructField("_kinesis_partition_key", StringType(), nullable=True),
        StructField("_kinesis_sequence_number", StringType(), nullable=True),
        StructField("_kinesis_approximate_arrival", TimestampType(), nullable=True),
    ]
    + BRONZE_AUDIT_FIELDS
)

BRONZE_DOCUMENTS_SCHEMA = StructType(
    [
        StructField("document_id", StringType(), nullable=False),
        StructField("path", StringType(), nullable=False),
        StructField("file_name", StringType(), nullable=True),
        StructField("file_extension", StringType(), nullable=True),
        StructField("modification_time", TimestampType(), nullable=True),
        StructField("length", LongType(), nullable=True),
        StructField("content", StringType(), nullable=True),
        StructField("content_sha256", StringType(), nullable=True),
    ]
    + BRONZE_AUDIT_FIELDS
)

# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------

SILVER_DOCUMENT_CHUNKS_SCHEMA = StructType(
    [
        StructField("chunk_id", StringType(), nullable=False),
        StructField("document_id", StringType(), nullable=False),
        StructField("chunk_index", IntegerType(), nullable=False),
        StructField("chunk_text", StringType(), nullable=False),
        StructField("char_count", IntegerType(), nullable=False),
        StructField("token_estimate", IntegerType(), nullable=False),
        StructField("source_path", StringType(), nullable=True),
        StructField("file_extension", StringType(), nullable=True),
        StructField("document_sha256", StringType(), nullable=True),
        StructField("language", StringType(), nullable=True),
        StructField("_processed_at", TimestampType(), nullable=False),
        StructField("_pipeline_run_id", StringType(), nullable=False),
    ]
)

SILVER_MARKET_EVENTS_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), nullable=False),
        StructField("event_type", StringType(), nullable=False),
        StructField("symbol", StringType(), nullable=False),
        StructField("occurred_at", TimestampType(), nullable=False),
        StructField("event_date", StringType(), nullable=False),
        StructField("price", DoubleType(), nullable=True),
        StructField("volume", LongType(), nullable=True),
        StructField("bid", DoubleType(), nullable=True),
        StructField("ask", DoubleType(), nullable=True),
        StructField("spread", DoubleType(), nullable=True),
        StructField("mid_price", DoubleType(), nullable=True),
        StructField("venue", StringType(), nullable=True),
        StructField("source_system", StringType(), nullable=True),
        StructField("_processed_at", TimestampType(), nullable=False),
        StructField("_pipeline_run_id", StringType(), nullable=False),
    ]
)

#: SCD Type 2 dimension. ``valid_from`` / ``valid_to`` / ``is_current`` are the
#: only three columns the merge logic in :mod:`pipelines.common.scd2` cares
#: about; everything else is business payload.
SILVER_INSTRUMENT_SCD2_SCHEMA = StructType(
    [
        StructField("instrument_key", StringType(), nullable=False),
        StructField("symbol", StringType(), nullable=False),
        StructField("issuer_name", StringType(), nullable=True),
        StructField("sector", StringType(), nullable=True),
        StructField("country", StringType(), nullable=True),
        StructField("currency", StringType(), nullable=True),
        StructField("listing_venue", StringType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
        # SCD2 control columns
        StructField("valid_from", TimestampType(), nullable=False),
        StructField("valid_to", TimestampType(), nullable=True),
        StructField("is_current", BooleanType(), nullable=False),
        StructField("row_hash", StringType(), nullable=False),
        StructField("_processed_at", TimestampType(), nullable=False),
        StructField("_pipeline_run_id", StringType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

GOLD_DOCUMENT_EMBEDDINGS_SCHEMA = StructType(
    [
        StructField("chunk_id", StringType(), nullable=False),
        StructField("document_id", StringType(), nullable=False),
        StructField("chunk_index", IntegerType(), nullable=False),
        StructField("chunk_text", StringType(), nullable=False),
        StructField("symbol", StringType(), nullable=True),
        StructField("sector", StringType(), nullable=True),
        StructField("source_path", StringType(), nullable=True),
        StructField("published_date", StringType(), nullable=True),
        StructField("embedding", ArrayType(FloatType()), nullable=True),
        StructField("_processed_at", TimestampType(), nullable=False),
    ]
)

GOLD_SYMBOL_DAILY_METRICS_SCHEMA = StructType(
    [
        StructField("symbol", StringType(), nullable=False),
        StructField("event_date", StringType(), nullable=False),
        StructField("sector", StringType(), nullable=True),
        StructField("event_count", LongType(), nullable=False),
        StructField("total_volume", LongType(), nullable=True),
        StructField("vwap", DoubleType(), nullable=True),
        StructField("open_price", DoubleType(), nullable=True),
        StructField("close_price", DoubleType(), nullable=True),
        StructField("high_price", DoubleType(), nullable=True),
        StructField("low_price", DoubleType(), nullable=True),
        StructField("avg_spread", DoubleType(), nullable=True),
        StructField("_processed_at", TimestampType(), nullable=False),
    ]
)

#: Rows rejected by the quality gate. One quarantine table per pipeline keeps
#: the failure reason attached to the record that caused it.
QUARANTINE_AUDIT_SCHEMA = StructType(
    [
        StructField("_quarantine_reason", StringType(), nullable=False),
        StructField("_quarantined_at", TimestampType(), nullable=False),
        StructField("_pipeline_name", StringType(), nullable=False),
        StructField("_pipeline_run_id", StringType(), nullable=False),
    ]
)
