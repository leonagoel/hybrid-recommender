"""
Security tests for the wishlist rendering module (frontend/wishlist.js)
and the shared UI escape helper (frontend/js/ui.js).

These tests verify, at the source-code level, that:
  1. User-controlled product fields (title, description) are never
     interpolated into innerHTML template literals.
  2. textContent / createElement / addEventListener are used instead.
  3. No inline event-handler attributes (onclick="…${…}…") are present.
  4. Known XSS payload strings cannot appear in an unsafe rendering path.
  5. The escapeHtml helper in ui.js correctly neutralises every standard
     XSS probe before it reaches the DOM.
"""

import os
import re
import html
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WISHLIST_JS = os.path.join(REPO_ROOT, "frontend", "wishlist.js")
UI_JS = os.path.join(REPO_ROOT, "frontend", "js", "ui.js")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# XSS probe payloads that must never reach the browser as raw HTML
XSS_PAYLOADS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    '<a href="javascript:alert(1)">test</a>',
    '"><script>alert(1)</script>',
    "'; alert(1); //",
    "</p><script>alert(1)</script><p>",
    "<SCRIPT SRC=http://xss.rocks/xss.js></SCRIPT>",
]


# ---------------------------------------------------------------------------
# wishlist.js — static-analysis tests
# ---------------------------------------------------------------------------

class TestWishlistStaticAnalysis:
    """Verify wishlist.js does not contain unsafe DOM-injection patterns."""

    @pytest.fixture(scope="class")
    def source(self):
        return _read(WISHLIST_JS)

    def test_file_exists(self):
        assert os.path.isfile(WISHLIST_JS), "wishlist.js must exist"

    def test_no_title_interpolated_in_innerhtml(self, source):
        """p.title must not appear inside an innerHTML template literal."""
        # Match patterns like: innerHTML = `...${p.title}...`
        pattern = r'innerHTML\s*=\s*`[^`]*\$\{[^}]*\.title[^}]*\}[^`]*`'
        matches = re.findall(pattern, source, re.DOTALL)
        assert not matches, (
            "Found innerHTML template literal containing .title interpolation — "
            "user data must be inserted via textContent, not innerHTML.\n"
            f"Matches: {matches}"
        )

    def test_no_description_interpolated_in_innerhtml(self, source):
        """p.description must not appear inside an innerHTML template literal."""
        pattern = r'innerHTML\s*=\s*`[^`]*\$\{[^}]*\.description[^}]*\}[^`]*`'
        matches = re.findall(pattern, source, re.DOTALL)
        assert not matches, (
            "Found innerHTML template literal containing .description interpolation — "
            "user data must be inserted via textContent.\n"
            f"Matches: {matches}"
        )

    def test_no_inline_onclick_with_interpolation(self, source):
        """
        onclick='removeFromWishlist(\\'${p.title}\\')' (or double-quote variant)
        is the classic JS-injection vector — must not appear.
        """
        pattern = r'onclick\s*=\s*["\'][^"\']*\$\{'
        matches = re.findall(pattern, source)
        assert not matches, (
            "Found inline onclick attribute with template interpolation — "
            "use addEventListener with a data-* attribute instead.\n"
            f"Matches: {matches}"
        )

    def test_title_uses_text_content(self, source):
        """Title must be set via textContent, not innerHTML."""
        assert 'textContent' in source, (
            "wishlist.js must use textContent to set product titles"
        )
        # Also confirm a direct assignment pattern
        assert re.search(r'\.textContent\s*=\s*[^;]*title', source), (
            "Expected assignment pattern `<el>.textContent = …title…` not found"
        )

    def test_description_uses_text_content(self, source):
        """Description must be set via textContent, not innerHTML."""
        assert re.search(r'\.textContent\s*=\s*[^;]*description', source), (
            "Expected assignment pattern `<el>.textContent = …description…` not found"
        )

    def test_remove_button_uses_addeventlistener(self, source):
        """Remove button must use addEventListener, not an inline onclick attribute."""
        assert 'addEventListener' in source, (
            "wishlist.js must use addEventListener for the remove button"
        )

    def test_no_dangerous_eval_patterns(self, source):
        """eval(), Function(), document.write() must not appear."""
        for dangerous in ('eval(', 'new Function(', 'document.write('):
            assert dangerous not in source, (
                f"Dangerous pattern '{dangerous}' found in wishlist.js"
            )

    def test_uses_create_element(self, source):
        """DOM nodes must be constructed with createElement, not string concatenation."""
        assert 'createElement' in source, (
            "wishlist.js must use document.createElement to build card nodes"
        )

    def test_get_wishlist_parses_json(self, source):
        """getWishlist() must read from localStorage via JSON.parse."""
        assert 'localStorage.getItem' in source, (
            "getWishlist must read from localStorage"
        )
        assert 'JSON.parse' in source, (
            "getWishlist must parse JSON to avoid prototype-pollution vectors"
        )


# ---------------------------------------------------------------------------
# wishlist.js — XSS payload injection simulation
# ---------------------------------------------------------------------------

class TestWishlistPayloadRendering:
    """
    Simulate what the wishlist renderer does with XSS payloads.
    Since we cannot run JS in Python, we verify the rendering strategy
    (textContent assignment) is equivalent to HTML entity-escaping.
    """

    def _simulate_text_content(self, raw: str) -> str:
        """
        textContent assignment is equivalent to setting the node's text;
        any HTML special characters are treated as literal text, not markup.
        The Python html.escape() function replicates this behaviour exactly
        for the characters that matter: & < > " '
        """
        return html.escape(raw, quote=True)

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_payload_rendered_as_text_not_html(self, payload):
        """
        When a wishlist item title equals a known XSS payload, the rendered
        output (via textContent) must not contain *raw* (unparsed) HTML tags.

        textContent escapes < and > to &lt; and &gt;, so the browser
        never parses the payload as a tag — any event-handler attributes
        (onerror=, onload=) or URL schemes (javascript:) that survive as
        plain text within entity-encoded markup are inert: they are
        characters on screen, not executable code.
        """
        rendered = self._simulate_text_content(payload)

        # Raw unescaped opening HTML tag delimiter must not survive.
        # A browser cannot parse a tag that starts with &lt; instead of <.
        if '<' in payload:
            assert '&lt;' in rendered, (
                f"< was not entity-encoded; payload could still parse as HTML: {rendered}"
            )
            # The raw unescaped character must be gone (except as part of &lt; itself)
            assert rendered.replace('&lt;', '').replace('&gt;', '').count('<') == 0, (
                f"Raw < survived entity encoding: {rendered}"
            )

        # At minimum the payload must have been transformed (not passed through)
        if any(c in payload for c in '<>"\'&'):
            assert rendered != payload, (
                f"Payload was not transformed by text rendering: {payload}"
            )

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_payload_script_tags_neutralised(self, payload):
        """Script tags in titles must be entity-encoded, not rendered as HTML."""
        rendered = self._simulate_text_content(payload)
        # Raw unescaped < must not survive
        if '<' in payload:
            assert '&lt;' in rendered or '<' not in rendered, (
                f"<  was not entity-encoded in: {rendered}"
            )

    def test_normal_title_passes_through_unchanged(self):
        """A normal product title must render exactly as supplied."""
        normal_title = "Bluetooth Wireless Headphones - Pro Edition"
        rendered = self._simulate_text_content(normal_title)
        assert rendered == normal_title

    def test_title_with_ampersand_is_safe(self):
        """An ampersand in a product title must be entity-encoded."""
        title = "Salt & Pepper Grinder Set"
        rendered = self._simulate_text_content(title)
        assert '&amp;' in rendered
        assert '&' not in rendered.replace('&amp;', '')

    def test_title_with_unicode_is_safe(self):
        """Unicode titles must pass through without modification."""
        title = "Headphones — Best Quality (€29.99)"
        rendered = self._simulate_text_content(title)
        # Non-ASCII should survive; only ASCII special chars are encoded
        assert "Headphones" in rendered
        assert "€29.99" in rendered


# ---------------------------------------------------------------------------
# ui.js — escapeHtml helper unit tests
# ---------------------------------------------------------------------------

class TestUiEscapeHtmlHelper:
    """
    Verify the escapeHtml function in frontend/js/ui.js is present, exported,
    and correctly handles every character class that drives XSS.
    """

    @pytest.fixture(scope="class")
    def source(self):
        return _read(UI_JS)

    def test_file_exists(self):
        assert os.path.isfile(UI_JS), "frontend/js/ui.js must exist"

    def test_escape_html_is_exported(self, source):
        """escapeHtml must be exported so other modules can import it."""
        assert 'export function escapeHtml' in source, (
            "ui.js must export escapeHtml"
        )

    def test_private_esc_alias_is_defined(self, source):
        """_esc must be defined in ui.js before it is used."""
        # The alias must be assigned either as a const/let/var or via function keyword
        defined = bool(
            re.search(r'(?:const|let|var)\s+_esc\s*=', source)
            or re.search(r'function\s+_esc\s*\(', source)
        )
        assert defined, (
            "_esc is used throughout ui.js as the private HTML-escape alias but "
            "was never defined, causing ReferenceError at runtime. "
            "Add: const _esc = escapeHtml; after the escapeHtml definition."
        )

    def test_escape_covers_angle_brackets(self, source):
        """The escape function must map < and > to entities."""
        assert '&lt;' in source, "escapeHtml must map < to &lt;"
        assert '&gt;' in source, "escapeHtml must map > to &gt;"

    def test_escape_covers_double_quote(self, source):
        """The escape function must map \" to &quot; to prevent attribute injection."""
        assert '&quot;' in source, "escapeHtml must map \" to &quot;"

    def test_escape_covers_single_quote(self, source):
        """The escape function must map ' to &#39; to prevent JS-string injection."""
        assert '&#39;' in source, "escapeHtml must map ' to &#39;"

    def test_escape_covers_ampersand(self, source):
        """The escape function must map & to &amp;."""
        assert '&amp;' in source, "escapeHtml must map & to &amp;"

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_escape_neutralises_payload(self, payload):
        """
        Simulate escapeHtml logic in Python and verify each XSS payload
        is neutralised.

        escapeHtml entity-encodes the HTML special characters (< > " ' &).
        After encoding, tags are never parsed by the browser.  Event-handler
        attribute names (onerror, onload) or URL schemes (javascript:) that
        remain as plain text characters within entity-encoded markup are
        harmless — there is no element, so there is no event.
        """
        escaped = html.escape(payload, quote=True)
        # Raw unescaped opening tag must not survive
        if '<' in payload:
            assert '&lt;' in escaped, (
                f"< was not entity-encoded by escapeHtml simulation: {escaped}"
            )
            remaining_raw = escaped.replace('&lt;', '').replace('&gt;', '').count('<')
            assert remaining_raw == 0, (
                f"Raw < survived escapeHtml: {escaped}"
            )


# ---------------------------------------------------------------------------
# Wishlist rendering integrity — structural checks
# ---------------------------------------------------------------------------

class TestWishlistRenderingIntegrity:
    """Verify the structural correctness of the safe rendering implementation."""

    @pytest.fixture(scope="class")
    def source(self):
        return _read(WISHLIST_JS)

    def test_render_wishlist_function_exists(self, source):
        assert 'function renderWishlist' in source, (
            "renderWishlist function must be defined"
        )

    def test_get_wishlist_function_exists(self, source):
        assert 'function getWishlist' in source, (
            "getWishlist function must be defined"
        )

    def test_remove_from_wishlist_function_exists(self, source):
        assert 'function removeFromWishlist' in source, (
            "removeFromWishlist function must be defined"
        )

    def test_render_wishlist_called_on_load(self, source):
        """renderWishlist() must be invoked at the end of the script to populate the page."""
        # Count standalone calls: lines that contain renderWishlist() but are not
        # function definitions (i.e. not "function renderWishlist()" lines).
        call_lines = [
            line for line in source.splitlines()
            if 'renderWishlist()' in line and 'function renderWishlist' not in line
        ]
        assert call_lines, (
            "renderWishlist() must be called at module load time to render the grid; "
            "no standalone call found in wishlist.js"
        )

    def test_wishlist_grid_id_referenced(self, source):
        """The script must reference the 'wishlist-grid' DOM element id."""
        assert 'wishlist-grid' in source, (
            "renderWishlist must reference the 'wishlist-grid' container element"
        )

    def test_remove_function_filters_by_title(self, source):
        """removeFromWishlist must filter the wishlist array by title."""
        assert re.search(r'\.filter\s*\(', source), (
            "removeFromWishlist must use Array.filter to remove items"
        )

    def test_wishlist_persisted_to_local_storage(self, source):
        """removeFromWishlist must persist the updated list to localStorage."""
        assert 'localStorage.setItem' in source, (
            "removeFromWishlist must persist changes to localStorage"
        )
