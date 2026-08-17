"""Bronze: unstructured document ingest with Databricks Auto Loader.

Reads whatever lands in the Unity Catalog volume (``/Volumes/<catalog>/bronze/
landing/...``) and appends it verbatim to ``bronze.raw_documents``. Bronze does
no parsing, no cleaning and no validation beyond "we can read the bytes" -- the
whole point of the layer is that it is a replayable record of what arrived.

Two Auto Loader details that matter in production:

``cloudFiles.useNotifications``
    Directory listing is O(files in the directory) per micro-batch and degrades
    badly past ~100k files. File notification mode (SQS/SNS) is O(new files).
    Start with listing, switch to notifications before the drop zone grows.

``cloudFiles.schemaEvolutionMode = rescue``
    A new or mistyped column lands in ``_rescued_data`` instead of failing the
    stream. For a Bronze layer this is exactly right: nothing is lost, nothing
    breaks, and the next Silver run can decide what to do about it.

Run:
    python -m pipelines.bronze.autoloader_documents --conf conf/dev.json
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..common.config import PipelineConfig
from ..common.logging_utils import configure_logging, stage
from ..common.spark import get_spark
from ..common.transforms import new_run_id, with_ingest_metadata

logger = logging.getLogger(__name__)

PIPELINE_NAME = "bronze_documents_autoloader"
STREAM_NAME = "bronze_documents"
TARGET_TABLE = "raw_documents"

#: Extensions Auto Loader will pick up. Anything else in the drop zone is
#: ignored rather than ingested as garbage.
SUPPORTED_GLOB = "*.{txt,md,json,html,htm,csv,pdf}"

#: Text-like formats are read as whole files; binary formats (PDF) keep their
#: bytes and are decoded downstream by the parsing job in Silver.
TEXT_EXTENSIONS = ("txt", "md", "html", "htm", "json", "csv")


def build_document_stream(spark, config: PipelineConfig) -> DataFrame:
    """Auto Loader source over the landing volume.

    ``binaryFile`` is the only format that handles PDFs and text uniformly, and
    it gives every row a path + modification time for free, which is what the
    ``document_id`` is derived from.
    """
    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.schemaLocation", config.checkpoint(f"{STREAM_NAME}/schema"))
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        # Bound the batch: without this the first run after a large backfill
        # tries to load the entire drop zone into one micro-batch.
        .option("cloudFiles.maxFilesPerTrigger", str(config.max_files_per_trigger))
        .option("cloudFiles.maxBytesPerTrigger", config.max_bytes_per_trigger)
        # Files are archived, never mutated in place; re-reading a modified
        # file would produce a second row with the same document_id.
        .option("cloudFiles.allowOverwrites", "false")
        .option("pathGlobFilter", SUPPORTED_GLOB)
        .option("recursiveFileLookup", "true")
    )

    if str(config.extra.get("use_file_notifications", "")).lower() in ("1", "true", "yes"):
        reader = reader.option("cloudFiles.useNotifications", "true").option(
            "cloudFiles.region", config.kinesis_region
        )

    return reader.load(config.landing_path)


def transform(df: DataFrame, *, run_id: str) -> DataFrame:
    """Derive identity and decode text, without interpreting the content."""
    file_name = F.element_at(F.split(F.col("path"), "/"), -1)
    extension = F.lower(F.regexp_extract(F.col("path"), r"\.([A-Za-z0-9]+)$", 1))

    # document_id is the content hash, not the path: the same document arriving
    # twice under two names is one document, and a re-upload of changed content
    # is a new one. That makes the whole downstream chain idempotent.
    content_sha = F.sha2(F.col("content"), 256)

    decoded = F.when(
        extension.isin(*TEXT_EXTENSIONS),
        F.decode(F.col("content"), "UTF-8"),
    ).otherwise(F.lit(None).cast("string"))

    enriched = (
        df.withColumn("file_name", file_name)
        .withColumn("file_extension", extension)
        .withColumn("content_sha256", content_sha)
        .withColumn("document_id", content_sha)
        .withColumn("content", decoded)
        .withColumnRenamed("modificationTime", "modification_time")
    )

    return with_ingest_metadata(
        enriched,
        pipeline_name=PIPELINE_NAME,
        run_id=run_id,
        source_file_col="path",
    ).select(
        "document_id",
        "path",
        "file_name",
        "file_extension",
        "modification_time",
        "length",
        "content",
        "content_sha256",
        "_ingest_timestamp",
        "_ingest_date",
        "_source_file",
        "_pipeline_run_id",
        "_pipeline_name",
    )


def run(config: PipelineConfig, *, await_termination: bool = True):
    spark = get_spark(PIPELINE_NAME)
    run_id = new_run_id()
    target = config.bronze(TARGET_TABLE)

    with stage(logger, PIPELINE_NAME, run_id=run_id, target=target):
        stream = transform(build_document_stream(spark, config), run_id=run_id)

        query = (
            stream.writeStream.format("delta")
            .outputMode("append")
            .option("checkpointLocation", config.checkpoint(STREAM_NAME))
            .option("mergeSchema", "true")
            # Ingest date is the only sensible partition key for an append-only
            # Bronze table: it matches how the data is replayed and expired.
            .partitionBy("_ingest_date")
            .queryName(STREAM_NAME)
            .trigger(processingTime=config.trigger_interval)
            .toTable(target)
        )

        logger.info(
            "stream.started",
            extra={"query_id": query.id, "target": target, "source": config.landing_path},
        )
        if await_termination:
            query.awaitTermination()
        return query


def main() -> None:
    configure_logging()
    run(PipelineConfig.resolve())


if __name__ == "__main__":
    main()
