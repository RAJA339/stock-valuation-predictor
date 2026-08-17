"""SCD Type 2 staging logic.

These tests cover the part that is easy to get wrong and hard to notice: which
rows the MERGE is asked to write. The Delta MERGE itself is exercised by the
integration job, not here -- what matters is that an unchanged row produces no
write, a changed row produces exactly two, and a new key produces exactly one.
"""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from pipelines.common.scd2 import (
    END_OF_TIME,
    MERGE_KEY_PREFIX,
    Scd2Error,
    Scd2Spec,
    build_staged_updates,
    merge_condition,
    prepare_source,
    row_hash,
)

SPEC = Scd2Spec(
    keys=["instrument_key"],
    tracked_columns=["symbol", "sector", "currency"],
)


@pytest.fixture
def source(spark):
    return spark.createDataFrame(
        [
            ("AAPL", "AAPL", "Technology", "USD"),
            ("MSFT", "MSFT", "Technology", "USD"),
            ("SHEL", "SHEL", "Energy", "GBP"),
        ],
        "instrument_key STRING, symbol STRING, sector STRING, currency STRING",
    )


def current_target(spark, rows):
    """Build an `is_current = true` target slice with matching row hashes."""
    df = spark.createDataFrame(
        rows, "instrument_key STRING, symbol STRING, sector STRING, currency STRING"
    )
    return (
        df.withColumn("row_hash", row_hash(SPEC.tracked_columns))
        .withColumn("valid_from", F.lit("2026-01-01 00:00:00").cast("timestamp"))
        .withColumn("valid_to", F.lit(END_OF_TIME).cast("timestamp"))
        .withColumn("is_current", F.lit(True))
    )


def merge_key_values(staged):
    col = f"{MERGE_KEY_PREFIX}instrument_key"
    return [r[col] for r in staged.collect()]


class TestScd2Spec:
    def test_rejects_empty_keys(self):
        with pytest.raises(Scd2Error, match="business key"):
            Scd2Spec(keys=[], tracked_columns=["a"])

    def test_rejects_empty_tracked_columns(self):
        with pytest.raises(Scd2Error, match="tracked column"):
            Scd2Spec(keys=["k"], tracked_columns=[])

    def test_key_cannot_also_be_a_tracked_attribute(self):
        # A changed key is a different entity, not a new version of this one.
        with pytest.raises(Scd2Error, match="must not be tracked"):
            Scd2Spec(keys=["k"], tracked_columns=["k", "a"])

    def test_control_columns_cannot_double_as_data(self):
        with pytest.raises(Scd2Error, match="control columns"):
            Scd2Spec(keys=["k"], tracked_columns=["is_current"])


class TestRowHash:
    def test_is_stable_for_identical_attributes(self, spark, source):
        a = source.withColumn("h", row_hash(SPEC.tracked_columns)).select("instrument_key", "h")
        b = source.withColumn("h", row_hash(SPEC.tracked_columns)).select("instrument_key", "h")
        assert {r.instrument_key: r.h for r in a.collect()} == {
            r.instrument_key: r.h for r in b.collect()
        }

    def test_changes_when_any_tracked_column_changes(self, spark, source):
        before = source.withColumn("h", row_hash(SPEC.tracked_columns))
        after = source.withColumn(
            "sector",
            F.when(F.col("instrument_key") == "AAPL", "Consumer").otherwise(F.col("sector")),
        ).withColumn("h", row_hash(SPEC.tracked_columns))

        before_map = {r.instrument_key: r.h for r in before.collect()}
        after_map = {r.instrument_key: r.h for r in after.collect()}
        assert before_map["AAPL"] != after_map["AAPL"]
        assert before_map["MSFT"] == after_map["MSFT"]

    def test_distinguishes_null_from_shifted_values(self, spark):
        # concat_ws would collapse ("a", NULL) and (NULL, "a") to the same
        # string; the NULL sentinel is what prevents a missed change.
        df = spark.createDataFrame([("a", None), (None, "a")], "x STRING, y STRING")
        hashes = [r.h for r in df.withColumn("h", row_hash(["x", "y"])).collect()]
        assert hashes[0] != hashes[1]

    def test_requires_at_least_one_column(self):
        with pytest.raises(Scd2Error):
            row_hash([])


class TestPrepareSource:
    def test_adds_control_columns(self, source):
        prepared = prepare_source(source, SPEC)
        for col in ("row_hash", "valid_from", "valid_to", "is_current"):
            assert col in prepared.columns
        row = prepared.first()
        assert row.is_current is True
        assert str(row.valid_to).startswith("9999-12-31")

    def test_deduplicates_on_the_business_key(self, spark):
        # Two rows for one key would trip DELTA_MULTIPLE_SOURCE_ROW_MATCHING.
        duplicated = spark.createDataFrame(
            [
                ("AAPL", "AAPL", "Technology", "USD"),
                ("AAPL", "AAPL", "Consumer", "USD"),
            ],
            "instrument_key STRING, symbol STRING, sector STRING, currency STRING",
        )
        prepared = prepare_source(duplicated, SPEC)
        assert prepared.count() == 1

    def test_rejects_a_source_missing_tracked_columns(self, spark):
        incomplete = spark.createDataFrame([("AAPL",)], "instrument_key STRING")
        with pytest.raises(Scd2Error, match="missing required columns"):
            prepare_source(incomplete, SPEC)


class TestBuildStagedUpdates:
    def test_unchanged_rows_produce_no_writes(self, spark, source):
        target = current_target(
            spark,
            [
                ("AAPL", "AAPL", "Technology", "USD"),
                ("MSFT", "MSFT", "Technology", "USD"),
                ("SHEL", "SHEL", "Energy", "GBP"),
            ],
        )
        staged = build_staged_updates(prepare_source(source, SPEC), target, SPEC)
        assert staged.count() == 0, "an idempotent re-run must not touch the dimension"

    def test_new_key_produces_a_single_insert(self, spark, source):
        target = current_target(
            spark,
            [("AAPL", "AAPL", "Technology", "USD"), ("MSFT", "MSFT", "Technology", "USD")],
        )
        staged = build_staged_updates(prepare_source(source, SPEC), target, SPEC)
        assert staged.count() == 1
        assert merge_key_values(staged) == [None], "a new key must take the INSERT branch"
        assert staged.first().instrument_key == "SHEL"

    def test_changed_row_produces_close_and_insert(self, spark, source):
        target = current_target(
            spark,
            [
                ("AAPL", "AAPL", "Consumer", "USD"),  # sector differs
                ("MSFT", "MSFT", "Technology", "USD"),
                ("SHEL", "SHEL", "Energy", "GBP"),
            ],
        )
        staged = build_staged_updates(prepare_source(source, SPEC), target, SPEC)
        rows = staged.collect()
        assert len(rows) == 2
        assert all(r.instrument_key == "AAPL" for r in rows)
        keys = sorted(merge_key_values(staged), key=lambda v: (v is not None, v))
        assert keys == [None, "AAPL"], "one row closes the old version, one opens the new"

    def test_mixed_batch_stages_the_right_number_of_rows(self, spark, source):
        target = current_target(
            spark,
            [
                ("AAPL", "AAPL", "Consumer", "USD"),  # changed  -> 2 rows
                ("MSFT", "MSFT", "Technology", "USD"),  # unchanged -> 0 rows
                ("XOM", "XOM", "Energy", "USD"),  # absent from source
            ],
        )
        # SHEL is new -> 1 row.
        staged = build_staged_updates(prepare_source(source, SPEC), target, SPEC)
        assert staged.count() == 3
        by_key = {}
        for row in staged.collect():
            by_key.setdefault(row.instrument_key, []).append(
                row[f"{MERGE_KEY_PREFIX}instrument_key"]
            )
        assert sorted(by_key) == ["AAPL", "SHEL"]
        assert sorted(by_key["AAPL"], key=lambda v: v is not None) == [None, "AAPL"]
        assert by_key["SHEL"] == [None]

    def test_empty_target_inserts_everything_once(self, spark, source):
        empty = current_target(spark, []).limit(0)
        staged = build_staged_updates(prepare_source(source, SPEC), empty, SPEC)
        assert staged.count() == 3
        assert merge_key_values(staged) == [None, None, None]

    def test_requires_a_prepared_source(self, spark, source):
        target = current_target(spark, [("AAPL", "AAPL", "Technology", "USD")])
        with pytest.raises(Scd2Error, match="call prepare_source first"):
            build_staged_updates(source, target, SPEC)

    def test_staged_rows_keep_the_source_columns(self, spark, source):
        target = current_target(spark, [("AAPL", "AAPL", "Consumer", "USD")])
        prepared = prepare_source(source, SPEC)
        staged = build_staged_updates(prepared, target, SPEC)
        assert set(prepared.columns).issubset(set(staged.columns))


class TestMergeCondition:
    def test_matches_on_merge_key_and_current_flag(self):
        condition = merge_condition(SPEC)
        assert "t.instrument_key = s.__merge_key_instrument_key" in condition
        # Without the is_current predicate the MERGE would match every
        # historical version of the key, not just the open one.
        assert "t.is_current = true" in condition

    def test_composite_key_ands_every_column(self):
        spec = Scd2Spec(keys=["a", "b"], tracked_columns=["v"])
        condition = merge_condition(spec)
        assert "t.a = s.__merge_key_a" in condition
        assert "t.b = s.__merge_key_b" in condition
        assert condition.count(" AND ") == 2
