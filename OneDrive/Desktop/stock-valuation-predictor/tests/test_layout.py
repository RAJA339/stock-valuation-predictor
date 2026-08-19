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


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))

    def chan(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


class TestNoBlackSurfaces:
    """
    The brand ground is the darkest thing in the app.

    'No black anywhere' is the design rule the navy palette exists to serve,
    and it is only true if every other surface sits *above* the ground in
    luminance. A future tweak that darkens a card or a border past #1C1C28
    reintroduces the black this replaced, so the rule is asserted rather than
    remembered.
    """

    SURFACES = ("SIDEBAR", "PANEL", "PANEL_HI", "GRID", "BORDER")

    def test_ground_is_the_brand_navy(self):
        assert theme.BG.upper() == "#1C1C28"

    def test_nothing_is_darker_than_the_ground(self):
        ground = _luminance(theme.BG)
        for name in self.SURFACES:
            assert _luminance(getattr(theme, name)) >= ground, \
                f"{name} is darker than the brand ground"

    def test_surfaces_stay_one_hue(self):
        """Blue above red and green on every surface — navy, never grey."""
        for name in ("BG",) + self.SURFACES:
            h = getattr(theme, name).lstrip("#")
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            assert b > max(r, g) + 5, f"{name} has lost its blue lift"

    def test_streamlit_chrome_is_pinned_to_the_ground(self):
        """Header and iframes paint Streamlit's near-black unless overridden."""
        for sel in ('header[data-testid="stHeader"]', "iframe", "html, body"):
            assert sel in theme.CSS, f"{sel} not covered by the stylesheet"


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
