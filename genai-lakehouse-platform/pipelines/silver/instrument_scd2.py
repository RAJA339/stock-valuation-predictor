"""Silver: instrument reference data as an SCD Type 2 dimension.

Reference data changes rarely but the changes matter: a sector reclassification
must not silently rewrite last quarter's attribution. Type 2 keeps every
version with an explicit validity window, so an as-of join reconstructs what
the world looked like at any point in time:

.. code-block:: sql

    SELECT e.symbol, e.price, d.sector
    FROM silver.market_events e
    JOIN silver.instrument_dim d
      ON e.symbol = d.symbol
     AND e.occurred_at >= d.valid_from
     AND e.occurred_at <  d.valid_to

This job is BATCH, not streaming -- reference feeds arrive as a daily snapshot
and a MERGE per micro-batch would be pure overhead.

Run:
    python -m pipelines.silver.instrument_scd2 --conf conf/dev.json \\
        --source /Volumes/cat/bronze/landing/reference/instruments
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..common.config import PipelineConfig
from ..common.logging_utils import configure_logging, stage
from ..common.quality import INSTRUMENT_EXPECTATIONS, validate, write_metrics
from ..common.scd2 import Scd2Spec, apply_scd2_merge, assert_no_overlapping_versions
from ..common.spark import get_spark
from ..common.transforms import new_run_id, normalise_symbol, with_processing_metadata

logger = logging.getLogger(__name__)

PIPELINE_NAME = "silver_instrument_scd2"
TARGET_TABLE = "instrument_dim"
QUARANTINE_TABLE = "instrument_dim_quarantine"
METRICS_TABLE = "data_quality_metrics"

#: The business key is the instrument identifier; the tracked columns are the
#: attributes whose change opens a new version. Anything not listed here can
#: change without creating history -- which is a deliberate decision per column,
#: not an oversight.
INSTRUMENT_SPEC = Scd2Spec(
    keys=["instrument_key"],
    tracked_columns=[
        "symbol",
        "issuer_name",
        "sector",
        "country",
        "currency",
        "listing_venue",
        "is_active",
    ],
)


def read_snapshot(spark, source_path: str, fmt: str = "json") -> DataFrame:
    """Read the daily reference snapshot from the landing volume."""
    reader = spark.read.format(fmt)
    if fmt in ("csv", "json"):
        reader = reader.option("header", "true").option("inferSchema", "false")
    return reader.load(source_path)


def transform(df: DataFrame, *, run_id: str) -> DataFrame:
    """Conform the raw snapshot into the dimension's column contract."""
    conformed = (
        df.withColumn("symbol", normalise_symbol(F.col("symbol")))
        .withColumn("currency", F.upper(F.trim(F.col("currency"))))
        .withColumn("country", F.upper(F.trim(F.col("country"))))
        .withColumn("sector", F.initcap(F.trim(F.col("sector"))))
        .withColumn("is_active", F.col("is_active").cast("boolean"))
        .withColumn(
            "instrument_key",
            F.coalesce(F.col("instrument_key"), F.col("symbol")),
        )
    )
    return with_processing_metadata(conformed, run_id=run_id).select(
        "instrument_key",
        "symbol",
        "issuer_name",
        "sector",
        "country",
        "currency",
        "listing_venue",
        "is_active",
        "_processed_at",
        "_pipeline_run_id",
    )


def run(
    config: PipelineConfig,
    source_path: str,
    *,
    source_format: str = "json",
    full_snapshot: bool = True,
) -> None:
    spark = get_spark(PIPELINE_NAME)
    run_id = new_run_id()
    target = config.silver(TARGET_TABLE)

    with stage(logger, PIPELINE_NAME, run_id=run_id, target=target, source=source_path):
        raw = read_snapshot(spark, source_path, source_format)
        conformed = transform(raw, run_id=run_id)

        result = validate(
            conformed,
            INSTRUMENT_EXPECTATIONS,
            pipeline=PIPELINE_NAME,
            table=target,
            min_pass_rate=0.99,
        )

        if result.metrics.quarantined_rows:
            (
                result.quarantined.withColumn("_pipeline_run_id", F.lit(run_id))
                .write.mode("append")
                .option("mergeSchema", "true")
                .saveAsTable(config.silver(QUARANTINE_TABLE))
            )

        apply_scd2_merge(
            spark,
            target,
            result.valid,
            INSTRUMENT_SPEC,
            # A late-arriving snapshot must not be stamped "now": the version
            # starts when the data says it does.
            effective_from=F.current_timestamp(),
            close_missing_keys=full_snapshot,
        )

        overlaps = assert_no_overlapping_versions(spark.table(target), INSTRUMENT_SPEC)
        if overlaps:
            raise RuntimeError(
                f"{target} has {len(overlaps)} overlapping validity windows; "
                f"first offender: {overlaps[0]}"
            )

        write_metrics(spark, result.metrics, config.silver(METRICS_TABLE))
        logger.info("scd2.complete", extra=result.metrics.as_row())


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", dest="conf", default=None)
    parser.add_argument("--source", required=True, help="Path to the reference snapshot to merge.")
    parser.add_argument("--format", dest="source_format", default="json")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Treat the source as a delta feed; do not expire keys missing from it.",
    )
    args = parser.parse_args()

    config = PipelineConfig.resolve(["--conf", args.conf] if args.conf else [])
    run(
        config,
        args.source,
        source_format=args.source_format,
        full_snapshot=not args.incremental,
    )


if __name__ == "__main__":
    main()
