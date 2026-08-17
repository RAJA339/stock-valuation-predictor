"""Declarative data quality gate.

An expectation is a SQL boolean expression that must hold for every row.
Three enforcement actions are supported, and choosing between them is a
business decision, not a technical one:

``WARN``
    Record the violation count, let the row through. Use for things you want
    to trend but that do not make the row wrong.
``QUARANTINE``
    Divert the row to a quarantine table with the reason attached. The
    default: bad rows are kept, visible, and replayable -- never dropped.
``FAIL``
    Abort the batch. Reserved for violations that mean the upstream contract
    itself is broken (a missing primary key, a schema-level invariant).

A NULL condition counts as a failure. "I could not evaluate this rule" is not
evidence that the row is fine.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class Action(str, Enum):
    WARN = "warn"
    QUARANTINE = "quarantine"
    FAIL = "fail"


class DataQualityError(RuntimeError):
    """Raised when an expectation with ``action=FAIL`` is violated."""

    def __init__(self, message: str, metrics: QualityMetrics) -> None:
        super().__init__(message)
        self.metrics = metrics


@dataclass(frozen=True)
class Expectation:
    """A named invariant expressed as a SQL predicate."""

    name: str
    condition: str
    action: Action = Action.QUARANTINE
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"expectation name must be alphanumeric/underscore, got {self.name!r}")
        if not self.condition.strip():
            raise ValueError(f"expectation {self.name!r} has an empty condition")
        # Accept a plain string for `action` so rule sets can be loaded as JSON.
        object.__setattr__(self, "action", Action(self.action))


@dataclass
class QualityMetrics:
    """Outcome of one validation pass, suitable for writing to an audit table."""

    pipeline: str
    table: str
    total_rows: int
    passed_rows: int
    quarantined_rows: int
    failures_by_rule: dict[str, int] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def pass_rate(self) -> float:
        """Fraction of rows that passed. An empty batch passes vacuously."""
        if self.total_rows == 0:
            return 1.0
        return self.passed_rows / self.total_rows

    def as_row(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "table": self.table,
            "total_rows": self.total_rows,
            "passed_rows": self.passed_rows,
            "quarantined_rows": self.quarantined_rows,
            "pass_rate": self.pass_rate,
            "failures_by_rule": json.dumps(self.failures_by_rule, sort_keys=True),
            "evaluated_at": self.evaluated_at,
        }

    def summary(self) -> str:
        worst = sorted(self.failures_by_rule.items(), key=lambda kv: -kv[1])[:5]
        detail = ", ".join(f"{name}={count}" for name, count in worst if count) or "none"
        return (
            f"{self.pipeline}/{self.table}: {self.passed_rows}/{self.total_rows} rows passed "
            f"({self.pass_rate:.2%}), {self.quarantined_rows} quarantined; violations: {detail}"
        )


@dataclass
class ValidationResult:
    valid: DataFrame
    quarantined: DataFrame
    metrics: QualityMetrics


_FAILURES_COL = "_dq_failures"


def _failure_flag(expectation: Expectation):
    """1 when the row violates the expectation, 0 otherwise (NULL counts as 1)."""
    return F.when(F.coalesce(F.expr(expectation.condition), F.lit(False)), F.lit(0)).otherwise(
        F.lit(1)
    )


def with_failure_column(df: DataFrame, expectations: Sequence[Expectation]) -> DataFrame:
    """Attach ``_dq_failures``: the array of violated blocking-rule names."""
    blocking = [e for e in expectations if e.action is not Action.WARN]
    if not blocking:
        return df.withColumn(_FAILURES_COL, F.array().cast("array<string>"))

    names = F.array(
        *[
            F.when(
                F.coalesce(F.expr(e.condition), F.lit(False)), F.lit(None).cast("string")
            ).otherwise(F.lit(e.name))
            for e in blocking
        ]
    )
    return df.withColumn(_FAILURES_COL, F.array_compact(names))


def validate(
    df: DataFrame,
    expectations: Sequence[Expectation],
    *,
    pipeline: str,
    table: str,
    min_pass_rate: float | None = None,
) -> ValidationResult:
    """Split ``df`` into passing and quarantined rows and collect metrics.

    A single Spark action is used for the counts: the aggregation below is the
    only job triggered, so calling this on a streaming micro-batch stays cheap.
    """
    if not expectations:
        empty = df.limit(0).withColumn("_quarantine_reason", F.lit(None).cast("string"))
        return ValidationResult(
            valid=df,
            quarantined=empty,
            metrics=QualityMetrics(pipeline, table, 0, 0, 0),
        )

    flagged = with_failure_column(df, expectations).cache()

    agg_exprs = [F.count(F.lit(1)).alias("__total")]
    agg_exprs += [F.sum(_failure_flag(e)).alias(f"__rule_{i}") for i, e in enumerate(expectations)]
    agg_exprs.append(
        F.sum(F.when(F.size(F.col(_FAILURES_COL)) > 0, 1).otherwise(0)).alias("__quarantined")
    )

    row = flagged.agg(*agg_exprs).collect()[0]
    total = int(row["__total"] or 0)
    quarantined_rows = int(row["__quarantined"] or 0)
    failures = {e.name: int(row[f"__rule_{i}"] or 0) for i, e in enumerate(expectations)}

    metrics = QualityMetrics(
        pipeline=pipeline,
        table=table,
        total_rows=total,
        passed_rows=total - quarantined_rows,
        quarantined_rows=quarantined_rows,
        failures_by_rule=failures,
    )

    valid = flagged.filter(F.size(F.col(_FAILURES_COL)) == 0).drop(_FAILURES_COL)
    quarantined = (
        flagged.filter(F.size(F.col(_FAILURES_COL)) > 0)
        .withColumn("_quarantine_reason", F.concat_ws(",", F.col(_FAILURES_COL)))
        .withColumn("_quarantined_at", F.current_timestamp())
        .withColumn("_pipeline_name", F.lit(pipeline))
        .drop(_FAILURES_COL)
    )

    fatal = {
        e.name: failures[e.name]
        for e in expectations
        if e.action is Action.FAIL and failures[e.name] > 0
    }
    if fatal:
        raise DataQualityError(
            f"{table}: fatal expectations violated: {fatal}. {metrics.summary()}", metrics
        )

    if min_pass_rate is not None and metrics.pass_rate < min_pass_rate:
        raise DataQualityError(
            f"{table}: pass rate {metrics.pass_rate:.2%} is below the required "
            f"{min_pass_rate:.2%}. {metrics.summary()}",
            metrics,
        )

    logger.info(metrics.summary())
    return ValidationResult(valid=valid, quarantined=quarantined, metrics=metrics)


def write_metrics(spark, metrics: QualityMetrics, audit_table: str) -> None:
    """Append one row of quality metrics to the audit table.

    Failures here are logged, never raised: losing an observability write must
    not take down a pipeline that otherwise succeeded.
    """
    try:
        row = metrics.as_row()
        (
            spark.createDataFrame([row])
            .write.mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(audit_table)
        )
    except Exception:  # pragma: no cover - depends on a live metastore
        logger.exception("Could not write quality metrics to %s", audit_table)


# ---------------------------------------------------------------------------
# Shared rule sets
# ---------------------------------------------------------------------------

MARKET_EVENT_EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation(
        name="event_id_present",
        condition="event_id IS NOT NULL AND length(trim(event_id)) > 0",
        action=Action.FAIL,
        description="Without an event id the record cannot be deduplicated.",
    ),
    Expectation(
        name="symbol_present",
        condition="symbol IS NOT NULL AND length(trim(symbol)) BETWEEN 1 AND 12",
        action=Action.QUARANTINE,
    ),
    Expectation(
        name="occurred_at_present",
        condition="occurred_at IS NOT NULL",
        action=Action.QUARANTINE,
    ),
    Expectation(
        name="occurred_at_not_in_future",
        condition="occurred_at <= current_timestamp() + INTERVAL 5 MINUTES",
        action=Action.QUARANTINE,
        description="Clock skew beyond 5 minutes means a broken producer.",
    ),
    Expectation(
        name="price_non_negative",
        condition="price IS NULL OR price >= 0",
        action=Action.QUARANTINE,
    ),
    Expectation(
        name="volume_non_negative",
        condition="volume IS NULL OR volume >= 0",
        action=Action.QUARANTINE,
    ),
    Expectation(
        name="spread_not_crossed",
        condition="bid IS NULL OR ask IS NULL OR ask >= bid",
        action=Action.WARN,
        description="Crossed books happen legitimately at the open; trend, do not block.",
    ),
    Expectation(
        name="payload_parsed",
        condition="_corrupt_record IS NULL",
        action=Action.QUARANTINE,
    ),
)

DOCUMENT_CHUNK_EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation(
        name="chunk_id_present",
        condition="chunk_id IS NOT NULL",
        action=Action.FAIL,
    ),
    Expectation(
        name="document_id_present",
        condition="document_id IS NOT NULL",
        action=Action.FAIL,
    ),
    Expectation(
        name="chunk_text_non_empty",
        condition="chunk_text IS NOT NULL AND length(trim(chunk_text)) >= 50",
        action=Action.QUARANTINE,
        description="Chunks shorter than ~50 chars retrieve noise.",
    ),
    Expectation(
        name="chunk_within_context_budget",
        condition="token_estimate <= 8192",
        action=Action.QUARANTINE,
        description="Longer than the embedding model's context window.",
    ),
    Expectation(
        name="chunk_index_non_negative",
        condition="chunk_index >= 0",
        action=Action.QUARANTINE,
    ),
)

INSTRUMENT_EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation(
        name="instrument_key_present",
        condition="instrument_key IS NOT NULL",
        action=Action.FAIL,
    ),
    Expectation(
        name="symbol_present",
        condition="symbol IS NOT NULL AND length(trim(symbol)) > 0",
        action=Action.QUARANTINE,
    ),
    Expectation(
        name="currency_is_iso4217",
        condition="currency IS NULL OR currency RLIKE '^[A-Z]{3}$'",
        action=Action.QUARANTINE,
    ),
)
