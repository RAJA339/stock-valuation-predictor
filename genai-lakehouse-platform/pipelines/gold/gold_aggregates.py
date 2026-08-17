"""Gold: business-level aggregates.

Two products:

``gold.symbol_daily_metrics``
    Per-symbol daily OHLC / VWAP / spread, joined to the SCD2 dimension
    **as of the event timestamp** so historical rows keep the sector the
    instrument had at the time.

``gold.document_context``
    The Silver chunks enriched with the metadata a RAG retriever filters on
    (symbol, sector, published date). This is the table the Vector Search
    index syncs from -- see :mod:`pipelines.gold.vector_index`.

Run:
    python -m pipelines.gold.gold_aggregates --conf conf/dev.json --date 2026-08-17
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..common.config import PipelineConfig
from ..common.logging_utils import configure_logging, stage
from ..common.spark import get_spark
from ..common.transforms import new_run_id, upsert_by_key

logger = logging.getLogger(__name__)

PIPELINE_NAME = "gold_aggregates"
DAILY_METRICS_TABLE = "symbol_daily_metrics"
DOCUMENT_CONTEXT_TABLE = "document_context"

#: Symbols mentioned in a document are extracted with a cheap ticker regex.
#: A production deployment swaps this for a proper NER model; the column
#: contract stays identical, which is the point of isolating it here.
_TICKER_RE = r"\$([A-Z]{1,5})\b"


def as_of_join(
    facts: DataFrame,
    dimension: DataFrame,
    *,
    fact_key: str,
    dim_key: str,
    fact_time: str,
    valid_from: str = "valid_from",
    valid_to: str = "valid_to",
) -> DataFrame:
    """Point-in-time join of facts against an SCD2 dimension.

    The window is half-open (``>= valid_from``, ``< valid_to``), which is what
    makes exactly one version match: a closed upper bound would match two rows
    at the instant one version ends and the next begins.
    """
    return facts.alias("f").join(
        dimension.alias("d"),
        on=(
            (F.col(f"f.{fact_key}") == F.col(f"d.{dim_key}"))
            & (F.col(f"f.{fact_time}") >= F.col(f"d.{valid_from}"))
            & (F.col(f"f.{fact_time}") < F.col(f"d.{valid_to}"))
        ),
        how="left",
    )


def build_symbol_daily_metrics(events: DataFrame, instruments: DataFrame) -> DataFrame:
    """Daily per-symbol market metrics with point-in-time sector attribution."""
    enriched = as_of_join(
        events,
        instruments,
        fact_key="symbol",
        dim_key="symbol",
        fact_time="occurred_at",
    ).select(
        F.col("f.symbol").alias("symbol"),
        F.col("f.event_date").alias("event_date"),
        F.col("f.occurred_at").alias("occurred_at"),
        F.col("f.price").alias("price"),
        F.col("f.volume").alias("volume"),
        F.col("f.spread").alias("spread"),
        F.col("d.sector").alias("sector"),
    )

    # VWAP must be volume-weighted, not a mean of prices. Rows with NULL volume
    # contribute to neither numerator nor denominator rather than being
    # silently treated as zero-volume trades.
    notional = F.sum(F.col("price") * F.col("volume"))
    traded_volume = F.sum(F.when(F.col("price").isNotNull(), F.col("volume")))

    ordered_open = F.first(F.col("price"), ignorenulls=True)
    ordered_close = F.last(F.col("price"), ignorenulls=True)

    return (
        enriched.sortWithinPartitions("symbol", "event_date", "occurred_at")
        .groupBy("symbol", "event_date")
        .agg(
            F.first(F.col("sector"), ignorenulls=True).alias("sector"),
            F.count(F.lit(1)).alias("event_count"),
            F.sum(F.col("volume")).alias("total_volume"),
            F.when(traded_volume > 0, notional / traded_volume).alias("vwap"),
            ordered_open.alias("open_price"),
            ordered_close.alias("close_price"),
            F.max(F.col("price")).alias("high_price"),
            F.min(F.col("price")).alias("low_price"),
            F.avg(F.col("spread")).alias("avg_spread"),
        )
        .withColumn("_processed_at", F.current_timestamp())
    )


def build_document_context(chunks: DataFrame, instruments: DataFrame) -> DataFrame:
    """Chunks + retrieval filters, ready to be indexed.

    Filter columns live in the same table as the text on purpose: Databricks
    Vector Search filters on the columns it syncs, so a metadata column that
    is not here cannot be used to narrow a similarity search.
    """
    # F.regexp_extract rather than an expr() string: inside a SQL literal the
    # backslashes would be consumed as SQL escapes ('\b' becomes a backspace
    # character), so the pattern silently stops matching anything.
    first_ticker = F.regexp_extract(F.col("chunk_text"), _TICKER_RE, 1)
    mentioned = chunks.withColumn(
        "symbol",
        # regexp_extract returns '' rather than NULL when nothing matches.
        F.when(F.length(first_ticker) > 0, first_ticker).otherwise(F.lit(None).cast("string")),
    )

    current_instruments = instruments.filter(F.col("is_current") == True).select(  # noqa: E712
        F.col("symbol").alias("__dim_symbol"), F.col("sector")
    )

    published = F.coalesce(
        F.regexp_extract(F.col("source_path"), r"(\d{4}-\d{2}-\d{2})", 1),
        F.date_format(F.col("_processed_at"), "yyyy-MM-dd"),
    )

    return (
        mentioned.join(
            current_instruments,
            mentioned["symbol"] == current_instruments["__dim_symbol"],
            how="left",
        )
        .withColumn("published_date", published)
        .select(
            "chunk_id",
            "document_id",
            "chunk_index",
            "chunk_text",
            "symbol",
            "sector",
            "source_path",
            "published_date",
            F.current_timestamp().alias("_processed_at"),
        )
    )


def run(config: PipelineConfig, *, event_date: str | None = None) -> None:
    spark = get_spark(PIPELINE_NAME)
    run_id = new_run_id()

    with stage(logger, PIPELINE_NAME, run_id=run_id, event_date=event_date or "all"):
        events = spark.table(config.silver("market_events"))
        if event_date:
            # Partition pruning: a full recompute of every day is a nightly
            # job, not an hourly one.
            events = events.filter(F.col("event_date") == event_date)

        instruments = spark.table(config.silver("instrument_dim"))

        daily = build_symbol_daily_metrics(events, instruments)
        upsert_by_key(
            spark,
            daily,
            config.gold(DAILY_METRICS_TABLE),
            keys=["symbol", "event_date"],
            partition_by=["event_date"],
        )
        logger.info("gold.written", extra={"table": DAILY_METRICS_TABLE})

        chunks = spark.table(config.silver("document_chunks"))
        context = build_document_context(chunks, instruments)
        upsert_by_key(spark, context, config.gold(DOCUMENT_CONTEXT_TABLE), keys=["chunk_id"])
        logger.info("gold.written", extra={"table": DOCUMENT_CONTEXT_TABLE})

        # Change Data Feed is a hard requirement for a Delta Sync vector index:
        # without it the index can only be rebuilt in full, never incrementally.
        spark.sql(
            f"ALTER TABLE {config.gold(DOCUMENT_CONTEXT_TABLE)} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", dest="conf", default=None)
    parser.add_argument("--date", dest="event_date", default=None, help="yyyy-MM-dd to rebuild.")
    args = parser.parse_args()

    config = PipelineConfig.resolve(["--conf", args.conf] if args.conf else [])
    run(config, event_date=args.event_date)


if __name__ == "__main__":
    main()
