"""Silver: Bronze documents -> validated, chunked, retrievable text.

Reads ``bronze.raw_documents`` as a stream and writes
``silver.document_chunks``: one row per retrievable passage, ready for the Gold
embedding/index job.

``foreachBatch`` is used rather than a plain append because each micro-batch
has to do three things atomically-ish: validate, MERGE (not append) into the
target, and quarantine what failed. Append-only would duplicate every chunk
whenever a batch is replayed after a driver restart.

Run:
    python -m pipelines.silver.documents_silver --conf conf/dev.json
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from ..common.config import PipelineConfig
from ..common.logging_utils import configure_logging, stage
from ..common.quality import DOCUMENT_CHUNK_EXPECTATIONS, validate, write_metrics
from ..common.spark import get_spark
from ..common.text import detect_language
from ..common.transforms import (
    chunk_documents,
    deduplicate,
    new_run_id,
    upsert_by_key,
    with_processing_metadata,
)

logger = logging.getLogger(__name__)

PIPELINE_NAME = "silver_document_chunks"
STREAM_NAME = "silver_document_chunks"
SOURCE_TABLE = "raw_documents"
TARGET_TABLE = "document_chunks"
QUARANTINE_TABLE = "document_chunks_quarantine"
METRICS_TABLE = "data_quality_metrics"

# The return type is a DataType instance, not the string "string": parsing a
# type from a string requires an active SparkContext, and this runs at import
# time -- before the job has built its session.
_detect_language_udf = F.udf(detect_language, StringType())


def transform(df: DataFrame, config: PipelineConfig, *, run_id: str) -> DataFrame:
    """Bronze documents -> Silver chunks."""
    # A document arriving twice (re-upload, checkpoint replay) is one document:
    # keep the earliest ingest so re-processing does not churn _processed_at.
    unique_docs = deduplicate(
        df.filter(F.col("content").isNotNull() & (F.length(F.col("content")) > 0)),
        keys=["document_id"],
        order_by=F.col("_ingest_timestamp").asc(),
    )

    chunks = chunk_documents(
        unique_docs,
        text_col="content",
        id_col="document_id",
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    enriched = (
        chunks.withColumn("language", _detect_language_udf(F.col("chunk_text")))
        .withColumnRenamed("path", "source_path")
        .withColumnRenamed("content_sha256", "document_sha256")
    )

    return with_processing_metadata(enriched, run_id=run_id).select(
        "chunk_id",
        "document_id",
        "chunk_index",
        "chunk_text",
        "char_count",
        "token_estimate",
        "source_path",
        "file_extension",
        "document_sha256",
        "language",
        "_processed_at",
        "_pipeline_run_id",
    )


def process_batch(batch_df: DataFrame, batch_id: int, config: PipelineConfig, run_id: str) -> None:
    """Validate then MERGE one micro-batch. Called by ``foreachBatch``."""
    spark = batch_df.sparkSession
    target = config.silver(TARGET_TABLE)

    silver = transform(batch_df, config, run_id=run_id)

    result = validate(
        silver,
        DOCUMENT_CHUNK_EXPECTATIONS,
        pipeline=PIPELINE_NAME,
        table=target,
        # Below 80% something systemic is wrong with extraction; stop rather
        # than quietly poison the index with garbage chunks.
        min_pass_rate=0.80,
    )

    upsert_by_key(spark, result.valid, target, keys=["chunk_id"])

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
            # Bronze is append-only, but a backfill or a GDPR delete rewrites
            # files; ignoring those changes keeps the stream from aborting.
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
