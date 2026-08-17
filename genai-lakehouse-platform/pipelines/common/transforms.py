"""Reusable DataFrame transforms shared by the medallion pipelines."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from .text import chunk_text as _chunk_text


def new_run_id() -> str:
    """Correlation id stamped on every row a single job run writes.

    Makes "which run produced this row" answerable without reading the Delta
    history, which matters when three streams write the same table.
    """
    return uuid.uuid4().hex


def with_ingest_metadata(
    df: DataFrame,
    *,
    pipeline_name: str,
    run_id: str,
    source_file_col: str | None = None,
) -> DataFrame:
    """Stamp the Bronze audit columns onto a DataFrame."""
    source_file = F.col(source_file_col) if source_file_col else F.lit(None).cast("string")
    return (
        df.withColumn("_ingest_timestamp", F.current_timestamp())
        .withColumn("_ingest_date", F.date_format(F.current_timestamp(), "yyyy-MM-dd"))
        .withColumn("_source_file", source_file)
        .withColumn("_pipeline_run_id", F.lit(run_id))
        .withColumn("_pipeline_name", F.lit(pipeline_name))
    )


def with_processing_metadata(df: DataFrame, *, run_id: str) -> DataFrame:
    """Stamp the Silver/Gold audit columns onto a DataFrame."""
    return df.withColumn("_processed_at", F.current_timestamp()).withColumn(
        "_pipeline_run_id", F.lit(run_id)
    )


def deduplicate(
    df: DataFrame,
    keys: Sequence[str],
    order_by: Column | Sequence[Column],
) -> DataFrame:
    """Keep one row per key, the winner decided by ``order_by``.

    At-least-once delivery is the norm for Kinesis and for Auto Loader after a
    checkpoint replay, so Silver must be idempotent on the natural key rather
    than trusting the source to be.
    """
    if not keys:
        raise ValueError("deduplicate requires at least one key column")
    orders = [order_by] if isinstance(order_by, Column) else list(order_by)
    window = Window.partitionBy(*[F.col(k) for k in keys]).orderBy(*orders)
    return (
        df.withColumn("__dedupe_rn", F.row_number().over(window))
        .filter(F.col("__dedupe_rn") == 1)
        .drop("__dedupe_rn")
    )


def normalise_symbol(col: Column) -> Column:
    """Upper-case, strip whitespace, and collapse an empty string to NULL.

    ``''`` and ``NULL`` mean the same thing here, and having both makes every
    downstream join and GROUP BY subtly wrong.
    """
    trimmed = F.upper(F.trim(col))
    return F.when(F.length(trimmed) == 0, F.lit(None).cast("string")).otherwise(trimmed)


def parse_timestamp(col: Column) -> Column:
    """Parse an ISO-8601 or epoch-millis timestamp without failing the batch.

    Unparseable values become NULL and are caught by the quality gate, which
    reports them, rather than by an exception that kills the stream.
    """
    as_iso = F.to_timestamp(col)
    as_epoch_millis = F.timestamp_millis(col.cast("long"))
    return F.coalesce(
        as_iso,
        F.when(col.rlike("^[0-9]{10,13}$"), as_epoch_millis),
    )


def add_spread_metrics(df: DataFrame) -> DataFrame:
    """Derive bid/ask spread and mid price, guarding against a zero mid."""
    mid = (F.col("bid") + F.col("ask")) / F.lit(2.0)
    return df.withColumn("spread", F.col("ask") - F.col("bid")).withColumn(
        "mid_price", F.when(mid > 0, mid).otherwise(F.lit(None).cast("double"))
    )


def chunk_documents(
    df: DataFrame,
    *,
    text_col: str = "content",
    id_col: str = "document_id",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> DataFrame:
    """Explode a document DataFrame into one row per chunk.

    A plain Python UDF is used rather than a pandas UDF because the chunker is
    string-manipulation bound: arrow serialisation would dominate the runtime
    for what is a handful of regex passes per document.
    """
    from pyspark.sql.types import (
        ArrayType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    chunk_struct = StructType(
        [
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk_text", StringType(), False),
            StructField("char_count", IntegerType(), False),
            StructField("token_estimate", IntegerType(), False),
        ]
    )

    @F.udf(returnType=ArrayType(chunk_struct))
    def _chunk(text):
        return [tuple(c) for c in _chunk_text(text, chunk_size, chunk_overlap)]

    exploded = df.withColumn("__chunk", F.explode(_chunk(F.col(text_col)))).select(
        "*",
        F.col("__chunk.chunk_index").alias("chunk_index"),
        F.col("__chunk.chunk_text").alias("chunk_text"),
        F.col("__chunk.char_count").alias("char_count"),
        F.col("__chunk.token_estimate").alias("token_estimate"),
    )

    # chunk_id must be deterministic: a re-run of the same document has to
    # produce the same ids or the vector index accumulates duplicates.
    return exploded.drop("__chunk").withColumn(
        "chunk_id",
        F.sha2(F.concat_ws("::", F.col(id_col), F.col("chunk_index").cast("string")), 256),
    )


def upsert_by_key(
    spark,
    df: DataFrame,
    target_table: str,
    keys: Sequence[str],
    *,
    partition_by: Sequence[str] = (),
) -> None:
    """Idempotent MERGE of a batch into a Delta table on ``keys``.

    Used from ``foreachBatch`` so that a micro-batch replayed after a failure
    updates rather than duplicates.
    """
    from delta.tables import DeltaTable

    if not spark.catalog.tableExists(target_table):
        writer = df.write.format("delta")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(target_table)
        return

    condition = " AND ".join(f"t.{k} <=> s.{k}" for k in keys)
    (
        DeltaTable.forName(spark, target_table)
        .alias("t")
        .merge(df.alias("s"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
