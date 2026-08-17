"""Shared pytest fixtures.

A single local SparkSession is shared by the whole session: starting a JVM
costs ~5 seconds, and a per-test session turns a 30-second suite into a
10-minute one.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession tuned for fast, deterministic tests."""
    pyspark = pytest.importorskip("pyspark", reason="pyspark is required for DataFrame tests")
    from pyspark.sql import SparkSession

    # Without this, workers can pick a different interpreter than the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    session = (
        SparkSession.builder.master("local[2]")
        .appName("genai-lakehouse-tests")
        # 200 shuffle partitions on a 5-row DataFrame is 200 empty tasks.
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    assert pyspark.__version__  # keeps the import referenced for linters
    yield session
    session.stop()


@pytest.fixture
def now() -> datetime:
    """A fixed instant safely in the past.

    Deliberately clock-relative rather than a hard-coded date: the market-event
    rules reject timestamps more than five minutes in the future, so a literal
    would silently start failing the suite the moment the wall clock passed it.
    Truncating to the second keeps a single test run deterministic.
    """
    return (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0, tzinfo=None)


@pytest.fixture
def market_events(spark, now):
    """A batch with one good row, one duplicate, and several bad ones."""
    from pyspark.sql.types import (
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    schema = StructType(
        [
            StructField("event_id", StringType()),
            StructField("event_type", StringType()),
            StructField("symbol", StringType()),
            StructField("occurred_at", TimestampType()),
            StructField("price", DoubleType()),
            StructField("volume", LongType()),
            StructField("bid", DoubleType()),
            StructField("ask", DoubleType()),
            StructField("venue", StringType()),
            StructField("source_system", StringType()),
            StructField("_corrupt_record", StringType()),
            StructField("_kinesis_approximate_arrival", TimestampType()),
        ]
    )

    rows = [
        # good
        ("e1", "trade", "aapl ", now, 190.5, 100, 190.4, 190.6, "xnas", "feed-a", None, now),
        # duplicate of e1, arrived later with a newer occurred_at
        (
            "e1",
            "trade",
            "AAPL",
            now + timedelta(seconds=5),
            191.0,
            120,
            190.9,
            191.1,
            "XNAS",
            "feed-a",
            None,
            now + timedelta(seconds=6),
        ),
        # good, different symbol
        ("e2", "quote", "MSFT", now, 410.0, 50, 409.9, 410.2, "XNAS", "feed-a", None, now),
        # negative price -> quarantine
        ("e3", "trade", "MSFT", now, -1.0, 10, None, None, "XNAS", "feed-a", None, now),
        # missing symbol -> quarantine
        ("e4", "trade", None, now, 10.0, 10, None, None, "XNAS", "feed-a", None, now),
        # corrupt payload -> quarantine
        ("e5", "trade", "TSLA", now, None, None, None, None, None, "feed-b", "{bad json", now),
        # crossed book -> warn only, still passes
        ("e6", "quote", "NVDA", now, 900.0, 5, 901.0, 899.0, "XNAS", "feed-a", None, now),
    ]
    return spark.createDataFrame(rows, schema)


@pytest.fixture
def instrument_snapshot(spark):
    """A small reference snapshot for the SCD2 tests."""
    from pyspark.sql.types import BooleanType, StringType, StructField, StructType

    schema = StructType(
        [
            StructField("instrument_key", StringType(), False),
            StructField("symbol", StringType(), False),
            StructField("issuer_name", StringType()),
            StructField("sector", StringType()),
            StructField("country", StringType()),
            StructField("currency", StringType()),
            StructField("listing_venue", StringType()),
            StructField("is_active", BooleanType()),
        ]
    )
    rows = [
        ("AAPL", "AAPL", "Apple Inc", "Technology", "US", "USD", "XNAS", True),
        ("MSFT", "MSFT", "Microsoft Corp", "Technology", "US", "USD", "XNAS", True),
        ("SHEL", "SHEL", "Shell plc", "Energy", "GB", "GBP", "XLON", True),
    ]
    return spark.createDataFrame(rows, schema)
