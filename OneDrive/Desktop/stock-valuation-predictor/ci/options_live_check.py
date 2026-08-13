"""
Real-network option-chain check (manual, NOT run in CI).

CI has no route to Yahoo, so it can only prove the live *code path* is correct
against stubbed responses. This script proves the live *connection* works, by
hitting the real endpoint. Run it from a machine with network access:

    python ci/options_live_check.py            # defaults to AAPL
    python ci/options_live_check.py MSFT NVDA

Exits non-zero if the chain did not come back live, printing the recorded
reason so a failure is diagnosable (no network / rate-limited / no listed
options) rather than just "it showed synthetic".
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import pandas as pd

from svp.data import options as OPT
from svp.models import derivatives as DV


def check_ticker(ticker: str) -> bool:
    print(f"\n=== {ticker} ===")

    expiries = OPT.list_expiries(ticker)
    print(f"expiries returned : {len(expiries)}  {expiries[:4]}{' ...' if len(expiries) > 4 else ''}")

    chain = OPT.get_option_chain(ticker, spot_fallback=0.0)
    print(f"source            : {chain.source}")
    if not chain.is_live:
        print(f"reason            : {chain.reason or '(none recorded)'}")
        print("RESULT            : FAILED — chain did not come back live")
        return False

    print(f"expiry            : {chain.expiry}")
    print(f"spot              : ${chain.spot:,.2f}")
    print(f"calls / puts      : {len(chain.calls)} / {len(chain.puts)}")

    ok = True

    if chain.spot <= 0:
        print("!! spot did not resolve — underlying and fast_info both empty")
        ok = False

    for side, df in (("calls", chain.calls), ("puts", chain.puts)):
        if df.empty:
            print(f"!! {side} frame is empty")
            ok = False
            continue
        missing = [c for c in OPT.CHAIN_COLUMNS if c not in df.columns]
        if missing:
            print(f"!! {side} missing columns: {missing}")
            ok = False
        if df["strike"].isna().any():
            print(f"!! {side} has NaN strikes")
            ok = False

    # Spot should sit inside the listed strike range for a normal chain.
    if not chain.calls.empty:
        lo, hi = chain.calls["strike"].min(), chain.calls["strike"].max()
        inside = lo <= chain.spot <= hi
        print(f"strike range      : ${lo:,.2f} – ${hi:,.2f}  (spot inside: {inside})")
        if not inside:
            print("!! spot outside the strike range — check the spot source")
            ok = False

    # Sanity-check live IV against a Black-Scholes round-trip on the near-ATM
    # call: a quoted premium should re-solve to roughly the quoted IV.
    atm = chain.calls.iloc[(chain.calls["strike"] - chain.spot).abs().argsort()[:1]]
    if not atm.empty:
        row = atm.iloc[0]
        mid = None
        if row["bid"] > 0 and row["ask"] > 0:
            mid = (row["bid"] + row["ask"]) / 2
        elif row["lastPrice"] > 0:
            mid = float(row["lastPrice"])
        T = max((pd.Timestamp(chain.expiry) - pd.Timestamp.today().normalize()).days, 1) / 365.0
        print(f"ATM strike        : ${row['strike']:,.2f}  premium ${mid if mid else float('nan'):,.2f}  "
              f"quoted IV {row['impliedVolatility'] * 100:,.1f}%")
        if mid and mid > 0:
            iv = DV.implied_volatility(mid, chain.spot, float(row["strike"]), T, 0.045)
            if iv is None:
                print("   (premium at/below intrinsic — IV not solvable, not necessarily an error)")
            else:
                print(f"solved IV         : {iv * 100:,.1f}%")
                if abs(iv - float(row["impliedVolatility"])) > 0.20:
                    print("!! solved IV differs from the quoted IV by >20 vol points")
                    ok = False

    print(f"RESULT            : {'PASSED' if ok else 'FAILED'}")
    return ok


def main():
    tickers = sys.argv[1:] or ["AAPL"]
    results = {t: check_ticker(t) for t in tickers}
    print("\n" + "=" * 46)
    for t, ok in results.items():
        print(f"  {t:<8} {'live OK' if ok else 'FAILED'}")
    if not all(results.values()):
        sys.exit(1)
    print("\nLIVE OPTION CHAIN CHECK PASSED")


if __name__ == "__main__":
    main()
