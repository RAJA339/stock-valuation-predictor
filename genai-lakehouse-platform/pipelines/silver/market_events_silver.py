"""Silver: Bronze market events -> conformed, deduplicated fact rows.

Everything that makes Bronze cheap to write (no types, no rules, duplicates
allowed) has to be paid for exactly once, here:

* types are enforced,
* symbols are normalised,
* at-least-once duplicates collapse on ``event_id``,
* rows that fail the contract are quarantined with the reason attached,
* derived measures (spread, mid price) are computed once for every consumer.

Run:
    python -m pipelines.silver.market_events_silver --conf conf/dev.json
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..common.config import PipelineConfig
from ..common.logging_utils import configure_logging, stage
from ..common.quality import MARKET_EVENT_EXPECTATIONS, validate, write_metrics
from ..common.spark import get_spark
from ..common.transforms import (
    add_spread_metrics,
    deduplicate,
    new_run_id,
    normalise_symbol,
    upsert_by_key,
    with_processing_metadata,
)

logger = logging.getLogger(__name__)

PIPELINE_NAME = "silver_market_events"
STREAM_NAME = "silver_market_events"
SOURCE_TABLE = "raw_market_events"
TARGET_TABLE = "market_events"
QUARANTINE_TABLE = "market_events_quarantine"
METRICS_TABLE = "data_quality_metrics"


def transform(df: DataFrame, *, run_id: str) -> DataFrame:
    """Bronze events -> Silver events."""
    conformed = (
        df.withColumn("symbol", normalise_symbol(F.col("symbol")))
        .withColumn("event_type", F.lower(F.trim(F.col("event_type"))))
        .withColumn("venue", F.upper(F.trim(F.col("venue"))))
        .withColumn("event_date", F.date_format(F.col("occurred_at"), "yyyy-MM-dd"))
    )

    # Kinesis guarantees at-least-once, so the same event_id can arrive on two
    # shards after a resharding event. The record that actually happened last
    # wins: order by the producer's timestamp, then by arrival as a tiebreak.
    unique = deduplicate(
        conformed,
        keys=["event_id"],
        order_by=[
            F.col("occurred_at").desc_nulls_last(),
            F.col("_kinesis_approximate_arrival").desc_nulls_last(),
        ],
    )

    with_metrics = add_spread_metrics(unique)

    return with_processing_metadata(with_metrics, run_id=run_id).select(
        "event_id",
        "event_type",
        "symbol",
        "occurred_at",
        "event_date",
        "price",
        "volume",
        "bid",
        "ask",
        "spread",
        "mid_price",
        "venue",
        "source_system",
        "_processed_at",
        "_pipeline_run_id",
    )


def process_batch(batch_df: DataFrame, batch_id: int, config: PipelineConfig, run_id: str) -> None:
    spark = batch_df.sparkSession
    target = config.silver(TARGET_TABLE)

    # The quality gate runs against the RAW batch so that corrupt records are
    # caught before transformation tries to make sense of them.
    result = validate(
        batch_df,
        MARKET_EVENT_EXPECTATIONS,
        pipeline=PIPELINE_NAME,
        table=target,
        min_pass_rate=0.95,
    )

    silver = transform(result.valid, run_id=run_id)
    upsert_by_key(spark, silver, target, keys=["event_id"], partition_by=["event_date"])

    if result.metrics.quarantined_rows:
        (
            result.quarantined.withColumn("_pipeline_run_id", F.lit(run_id))
            .write.mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(config.silver(QUARANTINE_TABLE))
        )

    write_metrics(spark, result.metrics, config.silver(METRICS_TABLE))
    logger.info("batch.complete", extra={"batch_id": batch_id, **result.metrics.as_row()})


def run(config: PipelineConfig, *, await_termination: bool = True):
    spark = get_spark(PIPELINE_NAME)
    run_id = new_run_id()
    source = config.bronze(SOURCE_TABLE)

    with stage(logger, PIPELINE_NAME, run_id=run_id, source=source):
        stream = (
            spark.readStream.format("delta")
            .option("ignoreChanges", "true")
            .option("maxFilesPerTrigger", str(config.max_files_per_trigger))
            .table(source)
        )

        query = (
            stream.writeStream.foreachBatch(
                lambda df, batch_id: process_batch(df, batch_id, config, run_id)
            )
            .option("checkpointLocation", config.checkpoint(STREAM_NAME))
            .queryName(STREAM_NAME)
            .trigger(processingTime=config.trigger_interval)
            .start()
        )

        logger.info("stream.started", extra={"query_id": query.id, "source": source})
        if await_termination:
            query.awaitTermination()
        return query


def main() -> None:
    configure_logging()
    run(PipelineConfig.resolve())


if __name__ == "__main__":
    main()
