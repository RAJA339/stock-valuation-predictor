"""
Pytest suite for the broker CSV importer.

The average-cost reconstruction is held to hand-computed cases — a sell must
remove cost at the running average, a split must cut average cost without
touching total cost — because a wrong average cost quietly poisons every P/L
figure downstream. The other property is honesty about the file: rows that
cannot move a share count are counted as skipped, and a file that is not a
broker CSV at all comes back as None, not as an empty portfolio.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.data import broker_import as BI       # noqa: E402

RH_HEADER = ('"Activity Date","Process Date","Settle Date","Instrument",'
             '"Description","Trans Code","Quantity","Price","Amount"')


def _rh(*rows):
    return RH_HEADER + "\n" + "\n".join(rows)


class TestRobinhoodTransactions:
    def test_single_buy(self):
        res = BI.parse_csv(_rh(
            '"1/10/2026","1/10/2026","1/12/2026","AAL","American Airlines",'
            '"Buy","100","$12.50","($1,250.00)"'))
        assert res is not None and res.source == "transactions"
        [p] = res.positions
        assert p.ticker == "AAL"
        assert p.shares == pytest.approx(100)
        assert p.avg_cost == pytest.approx(12.50)

    def test_two_lots_average(self):
        res = BI.parse_csv(_rh(
            '"2/01/2026","","","AAL","x","Buy","100","$20.00","x"',
            '"1/01/2026","","","AAL","x","Buy","100","$10.00","x"'))
        [p] = res.positions
        assert p.shares == pytest.approx(200)
        assert p.avg_cost == pytest.approx(15.00)

    def test_sell_removes_cost_at_the_running_average(self):
        # Buy 100 @ 10, buy 100 @ 20 → avg 15. Sell 100 → 100 left, avg still 15.
        res = BI.parse_csv(_rh(
            '"3/01/2026","","","AAL","x","Sell","100","$25.00","x"',
            '"2/01/2026","","","AAL","x","Buy","100","$20.00","x"',
            '"1/01/2026","","","AAL","x","Buy","100","$10.00","x"'))
        [p] = res.positions
        assert p.shares == pytest.approx(100)
        assert p.avg_cost == pytest.approx(15.00)

    def test_fully_closed_position_is_not_a_holding(self):
        res = BI.parse_csv(_rh(
            '"2/01/2026","","","AAL","x","Sell","100","$25.00","x"',
            '"1/01/2026","","","AAL","x","Buy","100","$10.00","x"'))
        assert res.positions == []

    def test_split_adds_shares_at_zero_cost(self):
        # 100 @ 20, then a 1:1 split credit of 100 → 200 shares, avg 10.
        res = BI.parse_csv(_rh(
            '"2/01/2026","","","TSLA","split","SPL","100","","x"',
            '"1/01/2026","","","TSLA","x","Buy","100","$20.00","x"'))
        [p] = res.positions
        assert p.shares == pytest.approx(200)
        assert p.avg_cost == pytest.approx(10.00)

    def test_dividends_and_junk_are_skipped_and_counted(self):
        res = BI.parse_csv(_rh(
            '"2/01/2026","","","AAL","dividend","CDIV","","","$12.00"',
            '"1/15/2026","","","","note row","","","",""',
            '"1/01/2026","","","AAL","x","Buy","10","$10.00","x"'))
        assert len(res.positions) == 1
        assert res.rows_skipped == 2
        assert any("skipped" in n for n in res.notes)

    def test_date_order_beats_file_order(self):
        """An oldest-first file must not be mis-read as newest-first."""
        res = BI.parse_csv(_rh(
            '"1/01/2026","","","AAL","x","Buy","100","$10.00","x"',
            '"2/01/2026","","","AAL","x","Buy","100","$20.00","x"',
            '"3/01/2026","","","AAL","x","Sell","150","$25.00","x"'))
        [p] = res.positions
        assert p.shares == pytest.approx(50)
        assert p.avg_cost == pytest.approx(15.00)

    def test_sell_of_unknown_shares_is_skipped_not_negative(self):
        res = BI.parse_csv(_rh(
            '"1/01/2026","","","AAL","x","Sell","100","$25.00","x"'))
        assert res.positions == []
        assert res.rows_skipped == 1

    def test_fractional_shares_survive(self):
        res = BI.parse_csv(_rh(
            '"1/01/2026","","","AAPL","x","Buy","0.3517","$190.10","x"'))
        [p] = res.positions
        assert p.shares == pytest.approx(0.3517)


class TestPositionsFormat:
    def test_generic_positions_export(self):
        res = BI.parse_csv(
            "Symbol,Shares,Average Cost\nAAL,100,12.50\nKO,25,\"$61.20\"\n")
        assert res.source == "positions"
        assert {p.ticker for p in res.positions} == {"AAL", "KO"}

    def test_duplicate_lots_merge_under_average(self):
        res = BI.parse_csv(
            "Ticker,Quantity,Avg Cost\nAAL,100,10.00\nAAL,100,20.00\n")
        [p] = res.positions
        assert p.shares == pytest.approx(200)
        assert p.avg_cost == pytest.approx(15.00)

    def test_bad_rows_skipped(self):
        res = BI.parse_csv(
            "Symbol,Shares,Average Cost\nAAL,100,12.50\n,5,1\nKO,0,10\nF,-3,10\n")
        assert len(res.positions) == 1
        assert res.rows_skipped == 3


class TestNotACSV:
    def test_garbage_is_none(self):
        assert BI.parse_csv("") is None
        assert BI.parse_csv("hello world\nthis is prose") is None

    def test_unrelated_csv_is_none(self):
        assert BI.parse_csv("a,b,c\n1,2,3\n") is None


class TestMoneyParsing:
    def test_num_handles_broker_formats(self):
        assert BI._num("$1,234.56") == pytest.approx(1234.56)
        assert BI._num("(45.10)") == pytest.approx(-45.10)
        assert BI._num("") is None
        assert BI._num("n/a") is None
