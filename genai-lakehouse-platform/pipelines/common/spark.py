"""SparkSession helpers.

On Databricks the session already exists and calling ``getOrCreate`` returns
it untouched. Locally (and in CI) we build a small Delta-enabled session so the
same pipeline code can be exercised without a cluster.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Settings that are wrong-by-default for a medallion workload.
_SESSION_CONF: dict[str, str] = {
    # Let AQE collapse the thousands of tiny files a streaming Bronze layer
    # produces instead of hard-coding shuffle partitions per job.
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    # Deletion vectors turn a MERGE from "rewrite the file" into "mark the
    # row"; on an SCD2 dimension this is the difference between minutes and
    # seconds per batch.
    "spark.databricks.delta.properties.defaults.enableDeletionVectors": "true",
    # A schema change should fail the job, not silently drop columns.
    "spark.databricks.delta.schema.autoMerge.enabled": "false",
    "spark.sql.sources.partitionOverwriteMode": "dynamic",
    "spark.sql.session.timeZone": "UTC",
}


def get_spark(app_name: str = "genai-lakehouse", **extra_conf: Any):
    """Return the active SparkSession, creating a local one if needed."""
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        for key, value in {**_SESSION_CONF, **extra_conf}.items():
            try:
                active.conf.set(key, value)
            except Exception:  # pragma: no cover - static confs on a cluster
                logger.debug("Could not set %s on the existing session", key)
        return active

    builder = SparkSession.builder.appName(app_name)
    for key, value in {**_SESSION_CONF, **extra_conf}.items():
        builder = builder.config(key, str(value))

    try:  # Delta is optional locally; pipelines that need it will say so.
        from delta import configure_spark_with_delta_pip

        builder = configure_spark_with_delta_pip(builder)
    except ImportError:  # pragma: no cover - exercised only without delta-spark
        logger.warning("delta-spark not installed; Delta-specific features disabled")

    return builder.getOrCreate()


def is_databricks(spark) -> bool:
    """True when running inside a Databricks runtime."""
    try:
        return spark.conf.get("spark.databricks.clusterUsageTags.clusterId", None) is not None
    except Exception:
        return False


def table_exists(spark, table_name: str) -> bool:
    """Catalog lookup that does not raise for a missing catalog or schema."""
    try:
        return spark.catalog.tableExists(table_name)
    except Exception:
        return False
