"""Slowly Changing Dimension Type 2 via a single Delta MERGE.

The problem MERGE cannot express directly: one changed source row needs TWO
writes against the target -- close the current version (UPDATE) and open a new
one (INSERT). A MERGE clause matches each source row at most once, so the
source is first expanded into "staged updates":

===========================  ==================  =========================
source row                   staged rows         effect
===========================  ==================  =========================
key exists, attrs changed    2 rows              1 closes the old version
                             (merge key set,     (matched -> UPDATE),
                             merge key NULL)     1 opens the new one
                                                 (unmatched -> INSERT)
key exists, attrs unchanged  0 rows              no write at all
key is new                   1 row               unmatched -> INSERT
===========================  ==================  =========================

The NULL merge key is doing the real work: ``t.key = s.__merge_key_key`` is
never true for NULL, so that row is guaranteed to take the INSERT branch.

:func:`build_staged_updates` is pure DataFrame algebra and is unit tested
without Delta; :func:`apply_scd2_merge` is the thin Delta wrapper around it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

#: Open-ended validity. A sentinel beats NULL for ``valid_to`` because
#: ``BETWEEN valid_from AND valid_to`` then works without a COALESCE, and
#: because a NULL end date is indistinguishable from "we forgot to set it".
END_OF_TIME = "9999-12-31 23:59:59"

MERGE_KEY_PREFIX = "__merge_key_"


class Scd2Error(ValueError):
    """Raised when the SCD2 inputs are structurally wrong."""


@dataclass(frozen=True)
class Scd2Spec:
    """Describes how a dimension tracks history."""

    keys: Sequence[str]
    tracked_columns: Sequence[str]
    valid_from_col: str = "valid_from"
    valid_to_col: str = "valid_to"
    is_current_col: str = "is_current"
    row_hash_col: str = "row_hash"

    def __post_init__(self) -> None:
        if not self.keys:
            raise Scd2Error("at least one business key is required")
        if not self.tracked_columns:
            raise Scd2Error("at least one tracked column is required")
        overlap = set(self.keys) & set(self.tracked_columns)
        if overlap:
            raise Scd2Error(
                f"business keys must not be tracked as attributes: {sorted(overlap)}. "
                "A changed key is a different entity, not a new version."
            )
        control = {
            self.valid_from_col,
            self.valid_to_col,
            self.is_current_col,
            self.row_hash_col,
        }
        clashes = control & (set(self.keys) | set(self.tracked_columns))
        if clashes:
            raise Scd2Error(f"control columns cannot double as data columns: {sorted(clashes)}")

    @property
    def merge_key_cols(self) -> list[str]:
        return [f"{MERGE_KEY_PREFIX}{k}" for k in self.keys]


def row_hash(columns: Sequence[str]) -> Column:
    """SHA-256 over the tracked attributes, NULL-safe and order-stable.

    NULLs are replaced with a sentinel that cannot appear in data, so
    ``("a", NULL)`` and ``(NULL, "a")`` hash differently -- ordinary
    ``concat_ws`` would collapse both to ``"a"`` and miss the change.
    """
    if not columns:
        raise Scd2Error("row_hash requires at least one column")
    parts: list[Column] = []
    for col in sorted(columns):
        parts.append(F.coalesce(F.col(col).cast("string"), F.lit("\u0000NULL")))
    return F.sha2(F.concat_ws("\u001f", *parts), 256)


def prepare_source(
    source: DataFrame,
    spec: Scd2Spec,
    *,
    effective_from: Column | None = None,
) -> DataFrame:
    """Add the SCD2 control columns to a batch of incoming records.

    The source is deduplicated on the business key, keeping the latest record
    by ``effective_from``. Feeding a MERGE two rows with the same key raises
    ``DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE`` at runtime, so
    this is a correctness requirement rather than an optimisation.
    """
    missing = [c for c in (*spec.keys, *spec.tracked_columns) if c not in source.columns]
    if missing:
        raise Scd2Error(f"source is missing required columns: {missing}")

    effective = effective_from if effective_from is not None else F.current_timestamp()

    prepared = (
        source.withColumn(spec.row_hash_col, row_hash(spec.tracked_columns))
        .withColumn(spec.valid_from_col, effective)
        .withColumn(spec.valid_to_col, F.lit(END_OF_TIME).cast("timestamp"))
        .withColumn(spec.is_current_col, F.lit(True))
    )

    from pyspark.sql import Window

    dedupe = Window.partitionBy(*[F.col(k) for k in spec.keys]).orderBy(
        F.col(spec.valid_from_col).desc()
    )
    return (
        prepared.withColumn("__rn", F.row_number().over(dedupe))
        .filter(F.col("__rn") == 1)
        .drop("__rn")
    )


def build_staged_updates(
    source: DataFrame,
    target_current: DataFrame,
    spec: Scd2Spec,
) -> DataFrame:
    """Expand ``source`` into the rows a single SCD2 MERGE needs.

    Args:
        source: output of :func:`prepare_source`.
        target_current: the ``is_current = true`` slice of the dimension.

    Returns:
        ``source`` columns plus one ``__merge_key_<key>`` column per business
        key. A NULL merge key marks a row that must be inserted as a new
        version; a populated one closes the existing version.
    """
    for col in (spec.row_hash_col, spec.valid_from_col):
        if col not in source.columns:
            raise Scd2Error(f"source is missing {col!r}; call prepare_source first")

    # The target side is projected to prefixed names before the join so the
    # result has no duplicate column names to disambiguate.
    target_probe = target_current.select(
        *[F.col(k).alias(f"__t_{k}") for k in spec.keys],
        F.col(spec.row_hash_col).alias("__t_row_hash"),
    )
    matched = source.join(
        target_probe,
        on=[F.col(k) == F.col(f"__t_{k}") for k in spec.keys],
        how="left",
    )

    is_new = F.col("__t_row_hash").isNull()
    is_changed = F.col("__t_row_hash").isNotNull() & (
        F.col("__t_row_hash") != F.col(spec.row_hash_col)
    )

    changed_or_new = matched.filter(is_new | is_changed)

    source_cols = list(source.columns)

    # Row A: closes the current version (only for changed keys).
    close_rows = changed_or_new.filter(is_changed).select(
        *source_cols,
        *[F.col(k).alias(f"{MERGE_KEY_PREFIX}{k}") for k in spec.keys],
    )

    # Row B: inserts the new version (for changed AND brand-new keys).
    insert_rows = changed_or_new.select(
        *source_cols,
        *[
            F.lit(None).cast(source.schema[k].dataType).alias(f"{MERGE_KEY_PREFIX}{k}")
            for k in spec.keys
        ],
    )

    return close_rows.unionByName(insert_rows)


def merge_condition(spec: Scd2Spec, target_alias: str = "t", source_alias: str = "s") -> str:
    """SQL ON-clause matching a staged row to the version it supersedes."""
    key_match = " AND ".join(
        f"{target_alias}.{k} = {source_alias}.{MERGE_KEY_PREFIX}{k}" for k in spec.keys
    )
    return f"{key_match} AND {target_alias}.{spec.is_current_col} = true"


def apply_scd2_merge(
    spark,
    target_table: str,
    source: DataFrame,
    spec: Scd2Spec,
    *,
    effective_from: Column | None = None,
    close_missing_keys: bool = False,
) -> DataFrame:
    """Merge ``source`` into ``target_table`` preserving Type 2 history.

    Args:
        close_missing_keys: when the source is a FULL snapshot, close every
            current row whose key is absent from it (a soft delete). Leave
            false for incremental feeds -- otherwise a partial batch expires
            the entire dimension.

    Returns:
        The staged-updates DataFrame that was merged, for logging/assertions.
    """
    from delta.tables import DeltaTable

    prepared = prepare_source(source, spec, effective_from=effective_from)

    if not spark.catalog.tableExists(target_table):
        logger.info("Creating SCD2 target %s from the first batch", target_table)
        (
            prepared.write.format("delta")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(target_table)
        )
        return prepared

    delta_table = DeltaTable.forName(spark, target_table)
    target_current = delta_table.toDF().filter(F.col(spec.is_current_col) == True)  # noqa: E712

    staged = build_staged_updates(prepared, target_current, spec).cache()

    update_set = {
        spec.is_current_col: F.lit(False),
        # The old version ends exactly where the new one begins: no gaps, no
        # overlaps, so an as-of join on a point in time hits exactly one row.
        spec.valid_to_col: F.col(f"s.{spec.valid_from_col}"),
    }

    (
        delta_table.alias("t")
        .merge(staged.alias("s"), merge_condition(spec))
        .whenMatchedUpdate(
            condition=f"t.{spec.row_hash_col} <> s.{spec.row_hash_col}",
            set=update_set,
        )
        .whenNotMatchedInsert(
            values={c: F.col(f"s.{c}") for c in prepared.columns},
        )
        .execute()
    )

    if close_missing_keys:
        _close_absent_keys(delta_table, prepared, spec)

    staged.unpersist()
    return staged


def _close_absent_keys(delta_table, snapshot: DataFrame, spec: Scd2Spec) -> None:
    """Expire current rows whose business key is missing from a full snapshot."""
    keys_present = snapshot.select(*spec.keys).distinct()
    current = delta_table.toDF().filter(F.col(spec.is_current_col) == True)  # noqa: E712
    to_close = current.join(keys_present, on=list(spec.keys), how="left_anti").select(
        *[F.col(k).alias(f"{MERGE_KEY_PREFIX}{k}") for k in spec.keys]
    )

    if to_close.isEmpty():
        return

    (
        delta_table.alias("t")
        .merge(to_close.alias("s"), merge_condition(spec))
        .whenMatchedUpdate(
            set={
                spec.is_current_col: F.lit(False),
                spec.valid_to_col: F.current_timestamp(),
            }
        )
        .execute()
    )


def assert_no_overlapping_versions(df: DataFrame, spec: Scd2Spec) -> list[dict]:
    """Return key ranges whose validity windows overlap. Empty means healthy.

    Worth running as a scheduled assertion: an overlap silently double-counts
    every as-of join against the dimension.
    """
    from pyspark.sql import Window

    window = Window.partitionBy(*spec.keys).orderBy(F.col(spec.valid_from_col))
    with_next = df.withColumn("__next_valid_from", F.lead(F.col(spec.valid_from_col)).over(window))
    bad = with_next.filter(
        F.col("__next_valid_from").isNotNull()
        & (F.col(spec.valid_to_col) > F.col("__next_valid_from"))
    ).select(*spec.keys, spec.valid_from_col, spec.valid_to_col, "__next_valid_from")
    return [row.asDict() for row in bad.limit(100).collect()]
