#!/usr/bin/env python3
"""
generate_annotated.py
=====================
Converts gemma4_simple.py into a labml.ai-style annotated HTML page.

Two-column layout:
  left (40%)  — prose / math from docstrings and inline # comments
  right (60%) — syntax-highlighted Python source

Usage:
    python tools/generate_annotated.py [--src gemma4_simple.py] [--out docs/index.html]

Annotation syntax (inside the Python source file):
  - Module / class / function docstrings   → doc panels
  - Consecutive  # lines before code       → doc panels
  - Section banners (# ─── Title ───)      → <h2> dividers spanning both columns
  - LaTeX math in docs: $...$  or  $$...$$
  - Markdown bold/italic/code works normally
"""
from __future__ import annotations

import argparse
import html
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import mistune
from pygments import highlight as pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


# ── Pygments setup ────────────────────────────────────────────────────────────

_PY_LEXER   = PythonLexer(stripnl=False)
_PY_FMT     = HtmlFormatter(nowrap=True, style="dracula", wrapcode=False)
_PY_CSS     = HtmlFormatter(style="dracula").get_style_defs(".code-hl")


# ── Mistune Markdown renderer with LaTeX pass-through ─────────────────────────

class _LatexSafeRenderer(mistune.HTMLRenderer):
    """Render markdown but leave $...$ / $$...$$ untouched for KaTeX."""

    def codespan(self, code: str) -> str:
        return f"<code>{html.escape(code)}</code>"


_MD = mistune.create_markdown(renderer=_LatexSafeRenderer())


def _render_md(text: str) -> str:
    """Render markdown, protecting LaTeX delimiters from HTML escaping."""
    # Temporarily replace $...$ with placeholders so mistune doesn't touch them
    placeholders: list[str] = []

    def _stash(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"\x00MATH{len(placeholders)-1}\x00"

    protected = re.sub(r"\$\$[^$]+?\$\$|\$[^$\n]+?\$", _stash, text, flags=re.DOTALL)
    rendered  = _MD(protected)
    for i, orig in enumerate(placeholders):
        rendered = rendered.replace(f"\x00MATH{i}\x00", orig)
    return rendered


# ── Section data model ────────────────────────────────────────────────────────

@dataclass
class Section:
    kind: str          # 'pair' | 'header' | 'code_only' | 'intro'
    docs: str  = ""    # raw markdown / prose text
    code: str  = ""    # raw Python source lines (may span multiple lines)
    lineno: int = 0    # first line number of the code part (1-based)


# ── Parser: source file → list[Section] ───────────────────────────────────────

_BANNER_RE   = re.compile(r"^#\s*[─━═]{4,}")          # section divider
_COMMENT_RE  = re.compile(r"^(\s*)#(?!!)(.*)$")        # any # line (not shebang)
_DOCSTR_RE   = re.compile(r'^\s*(r?""")')              # triple-quote open


def _strip_docstring(raw: str) -> str:
    """Remove enclosing triple quotes and dedent."""
    raw = raw.strip()
    for q in ('r"""', '"""', "r'''", "'''"):
        if raw.startswith(q):
            raw = raw[len(q):]
            break
    for q in ('"""', "'''"):
        if raw.endswith(q):
            raw = raw[:-len(q)]
            break
    return textwrap.dedent(raw).strip()


def _collect_docstring(lines: list[str], start: int) -> tuple[str, int]:
    """Read a triple-quoted docstring starting at `start`. Return (text, next_line)."""
    opening = lines[start]
    # Find the quote style
    m = re.search(r'(r?""")', opening)
    if not m:
        m = re.search(r"(r?''')", opening)
    if not m:
        return "", start + 1
    quote = '"""' if '"""' in m.group(0) else "'''"

    # Check if it closes on the same line (after the opening)
    after_open = opening[opening.index(m.group(0)) + len(m.group(0)):]
    if quote in after_open:
        # Single-line docstring
        raw = opening.strip()
        return _strip_docstring(raw), start + 1

    # Multi-line: collect until closing quote
    buf = [opening]
    i = start + 1
    while i < len(lines):
        buf.append(lines[i])
        if quote in lines[i]:
            break
        i += 1
    return _strip_docstring("".join(buf)), i + 1


def parse_source(source: str) -> list[Section]:
    lines = source.splitlines(keepends=True)
    sections: list[Section] = []
    i = 0

    def flush_code(code_buf: list[str], start: int, doc: str = "") -> None:
        code_text = "".join(code_buf).rstrip("\n")
        if not code_text.strip():
            return
        kind = "pair" if doc.strip() else "code_only"
        sections.append(Section(kind=kind, docs=doc, code=code_text, lineno=start))

    pending_doc  = ""          # accumulated docs text waiting for code
    code_buf: list[str] = []   # accumulated code lines for current section
    code_start   = 1
    comment_buf: list[str] = []  # consecutive # comment lines

    def flush_comments() -> str:
        """Convert accumulated comment lines to doc text."""
        nonlocal comment_buf
        if not comment_buf:
            return ""
        text = "\n".join(
            re.sub(r"^\s*#\s?", "", ln.rstrip()) for ln in comment_buf
        )
        comment_buf = []
        return text

    # ── Check for module-level docstring ──────────────────────────────────────
    # (first non-blank, non-shebang content)
    while i < len(lines) and lines[i].strip() in ("", "#!/usr/bin/env python3"):
        i += 1
    if i < len(lines) and _DOCSTR_RE.match(lines[i]):
        doc_text, i = _collect_docstring(lines, i)
        sections.append(Section(kind="intro", docs=doc_text, lineno=0))

    # ── Main pass ─────────────────────────────────────────────────────────────
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # ── Blank lines ───────────────────────────────────────────────────────
        if not stripped:
            if comment_buf:
                # Blank line ends a comment block; attach to next code chunk
                pending_doc = (pending_doc + "\n\n" + flush_comments()).strip()
            else:
                code_buf.append(raw)
            i += 1
            continue

        # ── Section banner pattern:
        #    # ──────────      (separator line)
        #    # Title text      (optional title comment)
        #    # ──────────      (separator line)
        if _BANNER_RE.match(stripped):
            flush_code(code_buf, code_start, pending_doc)
            code_buf = []
            pending_doc = ""
            comment_buf = []
            # Check if the next non-blank line is a title comment (not another banner)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            title = ""
            if j < len(lines):
                next_s = lines[j].strip()
                if next_s.startswith("#") and not _BANNER_RE.match(next_s):
                    title = re.sub(r"^\s*#\s*", "", next_s).strip()
                    i = j + 1   # skip the title comment line
                    # skip closing separator if present
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    if i < len(lines) and _BANNER_RE.match(lines[i].strip()):
                        i += 1
                else:
                    # Inline banner with title embedded: # ── Title ──
                    title = re.sub(r"#\s*[─━═]{2,}\s*", "", stripped).strip()
                    i += 1
            else:
                i += 1
            if title:
                sections.append(Section(kind="header", docs=title, lineno=i + 1))
            continue

        # ── Class / function definition ───────────────────────────────────────
        if re.match(r"^\s*(@|class |def |async def )", stripped):
            # Flush any pending code before this definition
            if code_buf:
                flush_code(code_buf, code_start, pending_doc)
                code_buf = []
                pending_doc = ""

            # Absorb comment block as doc for this definition
            pending_doc = (pending_doc + "\n\n" + flush_comments()).strip()

            def_line = raw
            def_start = i + 1
            i += 1

            # Collect decorator(s) and the actual def/class line
            while i < len(lines) and re.match(r"^\s*@", lines[i].strip()):
                def_line += lines[i]
                i += 1
            if i < len(lines) and not re.match(r"^\s*(class |def |async def )", def_line.strip()):
                def_line += lines[i]
                i += 1

            # Check for docstring immediately after
            doc_for_def = ""
            # Skip to first non-blank line inside the body
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and _DOCSTR_RE.match(lines[j]):
                doc_for_def, i = _collect_docstring(lines, j)

            combined_doc = (pending_doc + "\n\n" + doc_for_def).strip()
            sections.append(Section(
                kind="pair" if combined_doc else "code_only",
                docs=combined_doc,
                code=def_line.rstrip("\n"),
                lineno=def_start,
            ))
            pending_doc = ""
            code_start  = i + 1
            continue

        # ── Inline comment line ───────────────────────────────────────────────
        if _COMMENT_RE.match(stripped):
            # If code is accumulating, flush it first, then start comment block
            if code_buf and stripped.startswith("#"):
                flush_code(code_buf, code_start, pending_doc)
                code_buf = []
                pending_doc = ""
            comment_buf.append(raw)
            i += 1
            continue

        # ── Regular code line ─────────────────────────────────────────────────
        if comment_buf:
            # Comments just ended → they document the upcoming code
            pending_doc = (pending_doc + "\n\n" + flush_comments()).strip()
            code_start  = i + 1

        code_buf.append(raw)
        i += 1

    flush_code(code_buf, code_start, pending_doc)
    return sections


# ── HTML rendering ────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{
    delimiters: [
      {{left:'$$',right:'$$',display:true}},
      {{left:'$',right:'$',display:false}}
    ]
  }});"></script>
<style>
{css}
</style>
</head>
<body>
<nav><a href="#">{title}</a></nav>
{body}
<footer><p>Generated from <code>gemma4_simple.py</code></p></footer>
<script>{js}</script>
</body>
</html>
"""

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1d2127; color: #cdd; font-family: 'Segoe UI', system-ui, sans-serif;
       font-size: 15px; line-height: 1.65; }
nav { padding: 12px 24px; border-bottom: 1px solid #2e3440;
      font-size: 13px; color: #6c7a8a; }
nav a { color: #8fbcbb; text-decoration: none; }
footer { padding: 24px; text-align: center; color: #4c566a; font-size: 13px; }

/* ── Two-column section ── */
.section { display: flex; border-top: 1px solid #252a32; }
.section:first-child { border-top: none; }
.docs { width: 42%; padding: 20px 28px; background: #1d2127; color: #abb2bf;
        border-right: 1px solid #252a32; }
.code { width: 58%; padding: 16px 20px; background: #282c34; overflow-x: auto; }

/* ── Section header (spans both columns) ── */
.section-header { padding: 18px 28px 6px;
                  font-size: 12px; font-weight: 700; letter-spacing: .12em;
                  text-transform: uppercase; color: #5e81ac; border-top: 1px solid #252a32; }

/* ── Intro block (full width) ── */
.intro { max-width: 800px; margin: 32px auto; padding: 0 28px 32px;
         border-bottom: 1px solid #2e3440; }
.intro h1 { font-size: 24px; color: #eceff4; margin-bottom: 10px; }
.intro p, .intro li { color: #abb2bf; margin-bottom: 8px; }
.intro code { background: #2e3440; padding: 1px 5px; border-radius: 3px;
              font-size: 13px; color: #a3be8c; }

/* ── Docs typography ── */
.docs h2 { font-size: 14px; color: #d8dee9; margin-bottom: 8px; margin-top: 14px; }
.docs h3 { font-size: 13px; color: #b0bec5; margin-bottom: 6px; margin-top: 10px; }
.docs p { margin-bottom: 8px; font-size: 13.5px; }
.docs ul, .docs ol { padding-left: 18px; margin-bottom: 8px; font-size: 13.5px; }
.docs li { margin-bottom: 3px; }
.docs code { background: #2e3440; padding: 1px 5px; border-radius: 3px;
             font-size: 12px; color: #a3be8c; }
.docs strong { color: #d8dee9; }
.docs em { color: #88c0d0; font-style: italic; }
.docs .def-sig { font-family: monospace; font-size: 12px; color: #81a1c1;
                 background: #232830; padding: 4px 8px; border-radius: 4px;
                 display: block; margin-bottom: 10px; }

/* ── Code highlighting (Dracula) ── */
.code pre { margin: 0; font-family: 'JetBrains Mono','Fira Code',monospace; font-size: 13px; }
{pygments_css}

/* ── Line numbers ── */
.lineno { color: #495464; user-select: none; min-width: 3em;
          display: inline-block; text-align: right; margin-right: 12px; }

/* ── Interactive: lights-off on identifier click ── */
body.lights-off .code .highlight span:not(.lit) { color: #3a4050 !important; }
body.lights-off .code .highlight span.lit        { color: #00e5ff !important;
                                                   text-shadow: 0 0 14px #00e5ff88; }
/* ── Responsive ── */
@media (max-width: 760px) {
  .section { flex-direction: column; }
  .docs, .code { width: 100%; border-right: none; }
}
"""

_JS = r"""
// Click a Python identifier to spotlight all its occurrences ("lights off")
document.addEventListener('click', function(e) {
  var el = e.target;
  if (!el.closest('.code')) return;
  var tag = el.tagName.toLowerCase();
  // Pygments wraps identifiers in <span class="n"> / <span class="nf"> / <span class="nc"> etc.
  if (!el.classList.contains('n') && !el.classList.contains('nf') &&
      !el.classList.contains('nc') && !el.classList.contains('nb') &&
      !el.classList.contains('nn')) return;
  var name = el.textContent;
  // Remove previous highlights
  document.querySelectorAll('.code .highlight span.lit').forEach(function(s){
    s.classList.remove('lit');
  });
  if (document.body.classList.contains('lights-off') && _lastLit === name) {
    document.body.classList.remove('lights-off');
    _lastLit = null;
    return;
  }
  // Highlight all spans with the same text
  document.querySelectorAll('.code .highlight span').forEach(function(s) {
    if (s.textContent === name) s.classList.add('lit');
  });
  document.body.classList.add('lights-off');
  _lastLit = name;
});
var _lastLit = null;
"""


def _code_to_html(code: str, start_lineno: int) -> str:
    """Syntax-highlight `code` with line numbers starting at `start_lineno`."""
    highlighted = pyg_highlight(code, _PY_LEXER, _PY_FMT)
    # Add line numbers manually
    output_lines = []
    for rel, line in enumerate(highlighted.splitlines()):
        lno = start_lineno + rel
        output_lines.append(
            f'<span class="lineno">{lno}</span>{line}'
        )
    return '<pre class="highlight">' + "\n".join(output_lines) + "</pre>"


def _docs_to_html(docs: str) -> str:
    return _render_md(docs) if docs.strip() else ""


def render_html(sections: list[Section], title: str = "gemma4_simple.py") -> str:
    body_parts: list[str] = []

    for sec in sections:
        if sec.kind == "intro":
            content = _docs_to_html(sec.docs)
            body_parts.append(f'<div class="intro">{content}</div>')
        elif sec.kind == "header":
            body_parts.append(
                f'<div class="section-header">{html.escape(sec.docs)}</div>'
            )
        elif sec.kind in ("pair", "code_only"):
            docs_html = _docs_to_html(sec.docs)
            code_html = _code_to_html(sec.code, sec.lineno) if sec.code.strip() else ""
            body_parts.append(
                f'<div class="section">'
                f'<div class="docs">{docs_html}</div>'
                f'<div class="code">{code_html}</div>'
                f'</div>'
            )

    css = _CSS.replace("{pygments_css}", _PY_CSS)
    return _HTML_TEMPLATE.format(
        title=title,
        css=css,
        body="\n".join(body_parts),
        js=_JS,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="gemma4_simple.py",
                    help="Source Python file (default: gemma4_simple.py)")
    ap.add_argument("--out", default="docs/index.html",
                    help="Output HTML file (default: docs/index.html)")
    ap.add_argument("--title", default="Gemma 4 — Annotated Implementation",
                    help="Page title")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = src_path.read_text()
    sections = parse_source(source)
    html_out = render_html(sections, title=args.title)
    out_path.write_text(html_out)

    pair_count = sum(1 for s in sections if s.kind == "pair")
    header_count = sum(1 for s in sections if s.kind == "header")
    print(f"Generated {out_path}  ({len(sections)} sections, "
          f"{pair_count} annotated, {header_count} headers)")


if __name__ == "__main__":
    main()
