"""Bronze: Kinesis -> Delta via Structured Streaming.

The raw payload is preserved as a string alongside the parsed columns. That
redundancy is deliberate: when a producer changes its contract at 03:00, the
parsed columns go NULL but ``_raw_payload`` still holds the truth, and the
whole stream can be replayed from Bronze once the parser is fixed. Dropping
the raw bytes to save storage is a false economy.

Parsing uses ``PERMISSIVE`` mode with an explicit schema, so a malformed record
populates ``_corrupt_record`` and gets quarantined in Silver instead of killing
the stream.

Run:
    python -m pipelines.bronze.kinesis_market_events --conf conf/dev.json
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..common.config import PipelineConfig
from ..common.logging_utils import configure_logging, stage
from ..common.schemas import MARKET_EVENT_PAYLOAD_SCHEMA
from ..common.spark import get_spark
from ..common.transforms import new_run_id, parse_timestamp, with_ingest_metadata

logger = logging.getLogger(__name__)

PIPELINE_NAME = "bronze_market_events_kinesis"
STREAM_NAME = "bronze_market_events"
TARGET_TABLE = "raw_market_events"


def build_kinesis_stream(spark, config: PipelineConfig) -> DataFrame:
    """Kinesis source.

    ``TRIM_HORIZON`` on the first run replays everything still in the stream's
    retention window; ``LATEST`` starts at the tip and silently drops whatever
    was in flight. Default to TRIM_HORIZON and let a deliberate config change
    opt into data loss.
    """
    return (
        spark.readStream.format("kinesis")
        .option("streamName", config.kinesis_stream)
        .option("region", config.kinesis_region)
        .option("initialPosition", config.kinesis_initial_position)
        # Fetch interval and record limits keep a single micro-batch from
        # pulling a full shard's backlog after an outage.
        .option("maxRecordsPerFetch", "10000")
        .option("maxFetchRate", "1.0")
        .option("fetchBufferSize", "20gb")
        # Instance-profile credentials come from the ingest role created in
        # terraform/modules/iam; no keys are ever configured in code.
        .option("awsUseInstanceProfile", "true")
        .load()
    )


def transform(df: DataFrame, *, run_id: str) -> DataFrame:
    """Decode, parse and stamp the Kinesis records."""
    payload = F.col("data").cast("string")

    parsed = df.withColumn("_raw_payload", payload).withColumn(
        "__parsed",
        F.from_json(payload, MARKET_EVENT_PAYLOAD_SCHEMA, {"mode": "PERMISSIVE"}),
    )

    flattened = parsed.select(
        F.col("__parsed.event_id").alias("event_id"),
        F.col("__parsed.event_type").alias("event_type"),
        F.col("__parsed.symbol").alias("symbol"),
        parse_timestamp(F.col("__parsed.occurred_at")).alias("occurred_at"),
        F.col("__parsed.price").alias("price"),
        F.col("__parsed.volume").alias("volume"),
        F.col("__parsed.bid").alias("bid"),
        F.col("__parsed.ask").alias("ask"),
        F.col("__parsed.venue").alias("venue"),
        F.col("__parsed.source_system").alias("source_system"),
        F.col("__parsed.attributes").alias("attributes"),
        F.col("_raw_payload"),
        # from_json returns a NULL struct for unparseable JSON, so an
        # all-NULL parse is itself the corruption signal.
        F.coalesce(
            F.col("__parsed._corrupt_record"),
            F.when(F.col("__parsed").isNull(), F.col("_raw_payload")),
        ).alias("_corrupt_record"),
        F.col("partitionKey").alias("_kinesis_partition_key"),
        F.col("sequenceNumber").alias("_kinesis_sequence_number"),
        F.col("approximateArrivalTimestamp").alias("_kinesis_approximate_arrival"),
    )

    return with_ingest_metadata(flattened, pipeline_name=PIPELINE_NAME, run_id=run_id)


def run(config: PipelineConfig, *, await_termination: bool = True):
    spark = get_spark(PIPELINE_NAME)
    run_id = new_run_id()
    target = config.bronze(TARGET_TABLE)

    with stage(logger, PIPELINE_NAME, run_id=run_id, target=target):
        stream = transform(build_kinesis_stream(spark, config), run_id=run_id)

        query = (
            stream.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", config.checkpoint(STREAM_NAME))
            # Small-file pressure is the defining problem of a 30-second
            # trigger; optimized writes bin-pack each micro-batch on the way in
            # so OPTIMIZE has less to clean up later.
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact", "true")
            .partitionBy("_ingest_date")
            .queryName(STREAM_NAME)
            .trigger(processingTime=config.trigger_interval)
            .toTable(target)
        )

        logger.info(
            "stream.started",
            extra={
                "query_id": query.id,
                "target": target,
                "stream": config.kinesis_stream,
                "initial_position": config.kinesis_initial_position,
            },
        )
        if await_termination:
            query.awaitTermination()
        return query


def main() -> None:
    configure_logging()
    run(PipelineConfig.resolve())


if __name__ == "__main__":
    main()
