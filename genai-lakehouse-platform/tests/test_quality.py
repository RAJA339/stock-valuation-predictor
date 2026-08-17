"""Data quality gate behaviour."""

from __future__ import annotations

import pytest

from pipelines.common.quality import (
    MARKET_EVENT_EXPECTATIONS,
    Action,
    DataQualityError,
    Expectation,
    QualityMetrics,
    validate,
)


@pytest.fixture
def simple(spark):
    return spark.createDataFrame(
        [(1, "a", 10.0), (2, "b", -5.0), (3, None, 20.0), (4, "d", None)],
        "id INT, name STRING, amount DOUBLE",
    )


class TestExpectation:
    def test_rejects_empty_condition(self):
        with pytest.raises(ValueError, match="empty condition"):
            Expectation(name="x", condition="   ")

    def test_rejects_unsafe_name(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            Expectation(name="drop table; --", condition="1=1")

    def test_action_accepts_a_plain_string(self):
        # Rule sets are loaded from JSON, where the action is a string.
        assert Expectation(name="x", condition="1=1", action="fail").action is Action.FAIL


class TestValidate:
    def test_splits_valid_from_quarantined(self, simple):
        result = validate(
            simple,
            [Expectation(name="amount_positive", condition="amount > 0")],
            pipeline="t",
            table="t",
        )
        assert result.metrics.total_rows == 4
        assert result.metrics.passed_rows == 2  # -5.0 and NULL both fail
        assert result.metrics.quarantined_rows == 2
        assert sorted(r.id for r in result.valid.collect()) == [1, 3]

    def test_null_condition_counts_as_a_failure(self, simple):
        # `NULL > 0` is NULL, not false. Treating "cannot evaluate" as a pass
        # is how bad rows sail through a quality gate.
        result = validate(
            simple,
            [Expectation(name="amount_positive", condition="amount > 0")],
            pipeline="t",
            table="t",
        )
        quarantined_ids = sorted(r.id for r in result.quarantined.collect())
        assert 4 in quarantined_ids

    def test_quarantine_reason_lists_every_violated_rule(self, simple):
        result = validate(
            simple,
            [
                Expectation(name="amount_positive", condition="amount > 0"),
                Expectation(name="name_present", condition="name IS NOT NULL"),
            ],
            pipeline="t",
            table="t",
        )
        reasons = {r.id: r._quarantine_reason for r in result.quarantined.collect()}
        assert reasons[2] == "amount_positive"
        assert reasons[3] == "name_present"
        assert set(reasons[4].split(",")) == {"amount_positive"}

    def test_warn_action_does_not_quarantine(self, simple):
        result = validate(
            simple,
            [Expectation(name="amount_positive", condition="amount > 0", action=Action.WARN)],
            pipeline="t",
            table="t",
        )
        assert result.metrics.quarantined_rows == 0
        assert result.valid.count() == 4
        # ...but the violation is still counted, which is the whole point.
        assert result.metrics.failures_by_rule["amount_positive"] == 2

    def test_fail_action_aborts_the_batch(self, simple):
        with pytest.raises(DataQualityError, match="fatal expectations violated"):
            validate(
                simple,
                [
                    Expectation(
                        name="name_present", condition="name IS NOT NULL", action=Action.FAIL
                    )
                ],
                pipeline="t",
                table="t",
            )

    def test_min_pass_rate_is_enforced(self, simple):
        with pytest.raises(DataQualityError, match="pass rate"):
            validate(
                simple,
                [Expectation(name="amount_positive", condition="amount > 0")],
                pipeline="t",
                table="t",
                min_pass_rate=0.99,
            )

    def test_clean_batch_passes_entirely(self, simple):
        result = validate(
            simple,
            [Expectation(name="id_present", condition="id IS NOT NULL")],
            pipeline="t",
            table="t",
            min_pass_rate=1.0,
        )
        assert result.metrics.pass_rate == 1.0
        assert result.quarantined.count() == 0

    def test_empty_batch_passes_vacuously(self, spark, simple):
        result = validate(
            simple.limit(0),
            [Expectation(name="amount_positive", condition="amount > 0")],
            pipeline="t",
            table="t",
            min_pass_rate=1.0,
        )
        assert result.metrics.total_rows == 0
        assert result.metrics.pass_rate == 1.0

    def test_no_expectations_is_a_passthrough(self, simple):
        result = validate(simple, [], pipeline="t", table="t")
        assert result.valid.count() == 4


class TestMarketEventRules:
    def test_catches_each_defect_class(self, market_events):
        result = validate(
            market_events,
            [e for e in MARKET_EVENT_EXPECTATIONS if e.action is not Action.FAIL],
            pipeline="silver_market_events",
            table="silver.market_events",
        )
        quarantined = {r.event_id for r in result.quarantined.collect()}
        assert "e3" in quarantined, "negative price must be quarantined"
        assert "e4" in quarantined, "missing symbol must be quarantined"
        assert "e5" in quarantined, "corrupt payload must be quarantined"
        assert "e6" not in quarantined, "a crossed book is a warning, not a rejection"
        assert "e1" not in quarantined and "e2" not in quarantined

    def test_missing_event_id_is_fatal(self, spark, market_events):
        broken = market_events.withColumn(
            "event_id", market_events.event_id.substr(99, 1)
        )  # yields '' for every row
        with pytest.raises(DataQualityError):
            validate(
                broken,
                MARKET_EVENT_EXPECTATIONS,
                pipeline="p",
                table="silver.market_events",
            )


class TestQualityMetrics:
    def test_pass_rate_of_empty_batch_is_one(self):
        assert QualityMetrics("p", "t", 0, 0, 0).pass_rate == 1.0

    def test_summary_mentions_the_worst_rules(self):
        metrics = QualityMetrics("p", "t", 100, 90, 10, {"rule_a": 7, "rule_b": 3, "rule_c": 0})
        summary = metrics.summary()
        assert "rule_a=7" in summary
        assert "rule_c" not in summary  # zero-count rules are noise

    def test_as_row_serialises_failures_as_json(self):
        row = QualityMetrics("p", "t", 10, 9, 1, {"r": 1}).as_row()
        assert row["failures_by_rule"] == '{"r": 1}'
        assert row["pass_rate"] == 0.9
