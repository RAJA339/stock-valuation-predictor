"""
Pytest suite for the layout chrome: app header, footer, mobile CSS.

Thin by design — these are HTML/CSS strings, so the tests pin the contract
the entry script relies on: the header names the product and carries the
market status it was given, the footer carries the disclaimer verbatim, and
the stylesheet actually ships the classes and the mobile breakpoint the
markup references.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp import theme          # noqa: E402


class TestHeader:
    def test_carries_brand_status_and_stamp(self):
        h = theme.app_header("Open", True, "Rendered Aug 18, 10:30 AM ET")
        assert "INTRINSIC" in h
        assert "US market: Open" in h
        assert "Rendered Aug 18" in h
        assert "svp-dot-open" in h

    def test_closed_market_uses_the_muted_dot(self):
        h = theme.app_header("Closed", False, "stamp")
        assert "svp-dot-closed" in h and "svp-dot-open" not in h


class TestStreamlitThemeConfig:
    """
    .streamlit/config.toml must match the palette in svp/theme.py.

    Streamlit's native [theme] settings win over the injected CSS for the page
    background, so a palette change that updates theme.py alone leaves the app
    looking exactly as it did — which is precisely what happened when the
    ground moved to navy. A comment saying "keep the two in sync" is not a
    mechanism; this is.
    """

    @staticmethod
    def _config():
        import tomllib

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, ".streamlit", "config.toml"), "rb") as fh:
            return tomllib.load(fh)["theme"]

    def test_background_matches_the_palette_ground(self):
        assert self._config()["backgroundColor"].upper() == theme.BG.upper()

    def test_secondary_background_matches_the_card_surface(self):
        assert self._config()["secondaryBackgroundColor"].upper() == theme.PANEL.upper()

    def test_text_colour_matches(self):
        assert self._config()["textColor"].upper() == theme.TEXT.upper()

    def test_primary_matches_the_accent(self):
        assert self._config()["primaryColor"].upper() == theme.AMBER.upper()

    def test_base_is_dark(self):
        assert self._config()["base"] == "dark"


class TestFooterAndCSS:
    def test_footer_carries_the_disclaimer(self):
        assert "For educational purposes only" in theme.FOOTER_HTML
        assert "Not financial advice" in theme.FOOTER_HTML

    def test_css_ships_the_classes_the_markup_uses(self):
        for cls in (".svp-header", ".svp-footer", ".svp-brand",
                    ".svp-dot-open", ".svp-dot-closed"):
            assert cls in theme.CSS, f"{cls} missing from stylesheet"

    def test_css_has_a_mobile_breakpoint(self):
        assert "@media (max-width: 740px)" in theme.CSS

    def test_content_padding_clears_the_fixed_footer(self):
        assert "padding-bottom" in theme.CSS
