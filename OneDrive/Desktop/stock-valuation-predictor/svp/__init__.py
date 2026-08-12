"""
svp — Stock Valuation Predictor
================================

A modular toolkit powering the Intrinsic Stock Valuation Predictor Streamlit app.

Sub-packages
------------
- ``svp.data``      : live data pipelines (SEC EDGAR, yfinance, FRED/BLS, sentiment, storage)
- ``svp.models``    : ML valuation (point + quantile XGBoost), SHAP/LIME explainability,
                      DCF + Monte-Carlo scenario modelling, and the backtesting engine
- ``svp.analytics`` : peer-group benchmarking
- ``svp.features``  : feature engineering incl. QoQ / YoY trend & delta metrics
- ``svp.reports``   : one-click PDF equity research reports
- ``svp.theme``     : shared colour palette / CSS

Every module degrades gracefully: if an optional dependency (shap, lime, yfinance,
reportlab, transformers …) or a network API is unavailable, the relevant feature
falls back to a deterministic offline path instead of crashing the app.
"""

__version__ = "2.0.0"
__all__ = ["data", "models", "analytics", "features", "reports", "theme"]
