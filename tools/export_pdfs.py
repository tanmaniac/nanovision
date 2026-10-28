"""Export the assignment READMEs to phone-sized PDFs, with math and diagrams rendered.

Usage, from the repo root (`make pdf` or `make pdf A=a06_0_flow_matching` does the same):

    python tools/export_pdfs.py                       # every assignments/*/README.md
    python tools/export_pdfs.py a06_0_flow_matching   # one or more assignment ids

Each README goes through pandoc (GitHub-flavored markdown with $...$ math) to HTML, which headless
Chromium renders with KaTeX for the math and Mermaid for the diagrams, then prints to
pdf/<assignment>.pdf. Pages are 105 x 190 mm, close to a phone screen, so the text reads without
zooming. Collapsed <details> blocks (the exercise answers) are printed open. Display equations,
diagrams and tables wider than the page are scaled down to fit.

After each file the script checks the rendered page and reports problems: KaTeX parse errors,
Mermaid diagrams that failed to render, images that failed to load, '$' signs left in the text
(math pandoc did not recognize), and anything still wider than the page. It exits non-zero if any
file has a problem.

Needs pandoc on PATH and Playwright's Chromium:
    pip install playwright && python -m playwright install chromium-headless-shell
KaTeX and Mermaid load from cdn.jsdelivr.net, so the export needs network access.
"""

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pdf"

# Printable area of the 105 x 190 mm page with the margins in CSS, in CSS px (96 per inch).
PAGE_W_PX = round((105 - 12) / 25.4 * 96)
PAGE_H_PX = round((190 - 15) / 25.4 * 96)

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"

CSS = """
@page { size: 105mm 190mm; margin: 7mm 6mm 8mm 6mm; }
html { font-size: 10pt; }
body { font-family: "DejaVu Sans", "Helvetica", sans-serif; line-height: 1.45; color: #111;
       margin: 0; overflow-wrap: break-word; }
h1 { font-size: 1.55em; margin: 0 0 .5em; }
h2 { font-size: 1.3em; margin: 1.3em 0 .4em; border-bottom: 1px solid #ccc; padding-bottom: .1em; }
h3 { font-size: 1.12em; margin: 1.1em 0 .3em; }
h4 { font-size: 1em; margin: 1em 0 .3em; }
h2, h3, h4 { break-after: avoid; }
p, li { orphans: 2; widows: 2; }
img { max-width: 100%; height: auto; display: block; margin: .5em auto; break-inside: avoid; }
pre { white-space: pre-wrap; word-break: break-all; background: #f4f4f4; padding: .5em;
      border-radius: 4px; font-size: .8em; line-height: 1.3; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: .88em; }
pre code { font-size: 1em; }
table { border-collapse: collapse; font-size: .82em; margin: .6em 0; }
th, td { border: 1px solid #bbb; padding: .2em .35em; vertical-align: top; }
blockquote { margin: .6em 0; padding: 0 .7em; border-left: 3px solid #ccc; color: #444; }
details { border: 1px solid #ccc; border-radius: 4px; padding: .3em .6em; margin: .6em 0;
          background: #fafafa; }
summary { font-weight: bold; }
.katex-display { margin: .6em 0; break-inside: avoid; }
.katex { font-size: 1.05em; }
pre.mmd { background: none; text-align: center; white-space: pre; word-break: normal;
              break-inside: avoid; }
.wrapped-display { text-align: center; margin: .6em 0; line-height: 2; }
pre.mmd.rotated { break-before: page; break-after: page; margin: 0; padding: 0; }
"""

# Runs in the page after KaTeX and Mermaid are loaded. Renders everything, opens <details>,
# scales over-wide blocks, then returns a report of what went wrong.
RENDER_JS = r"""
async () => {
  document.querySelectorAll('details').forEach(d => d.open = true);
  document.querySelectorAll('span.math').forEach(el => {
    const display = el.classList.contains('display');
    try {
      katex.render(el.textContent, el, {displayMode: display, throwOnError: true,
                                        strict: 'ignore', trust: true});
    } catch (e) {
      el.innerHTML = '<span class="katex-error" style="color:#c00">' + el.textContent + '</span>';
      el.dataset.err = String(e.message || e);
    }
  });
  const blocks = [...document.querySelectorAll('pre.mmd')];
  mermaid.initialize({startOnLoad: false, securityLevel: 'loose', theme: 'default',
                      flowchart: {htmlLabels: true}});
  const mermaidErrors = [];
  for (const [i, pre] of blocks.entries()) {
    try {
      const {svg} = await mermaid.render('mmd' + i, pre.textContent);
      pre.innerHTML = svg;
      const s = pre.querySelector('svg');
      s.style.maxWidth = '100%'; s.style.height = 'auto';
    } catch (e) { mermaidErrors.push(String(e.message || e).slice(0, 200)); }
  }
  await Promise.all([...document.images].map(im => im.complete ? 0 :
      new Promise(r => { im.onload = im.onerror = r; })));
  const width = document.body.clientWidth;
  const pageH = PAGE_H;   // printable height in CSS px
  // A display equation wider than the page is first re-rendered so it can wrap after top-level
  // relations and operators (KaTeX breaks lines only outside display mode), then scaled if needed.
  for (const el of document.querySelectorAll('span.math.display')) {
    const d = el.querySelector('.katex-display');
    if (!d || d.scrollWidth <= width + 1) continue;
    const tex = el.querySelector('annotation')?.textContent;
    if (!tex) continue;
    const box = document.createElement('div');
    box.className = 'wrapped-display';
    katex.render('\\displaystyle ' + tex, box, {displayMode: false, throwOnError: false,
                                                strict: 'ignore', trust: true});
    el.replaceWith(box);
  }
  // Inline math cannot break inside a group, so one that runs past the right edge is made an
  // inline block, which moves it to the next line whole; it is scaled only if still too wide.
  for (const el of document.querySelectorAll('span.math.inline')) {
    if (el.getBoundingClientRect().right > width + 1) el.style.display = 'inline-block';
  }
  for (const el of document.querySelectorAll('.katex-display, .wrapped-display, table, span.math.inline')) {
    const w = el.scrollWidth;
    if (w > width + 1) { el.style.zoom = (width / w).toFixed(3); }
  }
  // A diagram that would shrink below 75% is turned sideways on its own page.
  for (const pre of document.querySelectorAll('pre.mmd')) {
    const svg = pre.querySelector('svg');
    if (!svg) continue;
    // Mermaid sets width="100%", so read the natural size from the viewBox.
    const vb = svg.viewBox.baseVal;
    const r = {width: vb.width, height: vb.height};
    const fit = width / r.width;
    if (fit >= 0.75) { svg.style.maxWidth = '100%'; continue; }
    // Nest the diagram in an outer SVG that rotates it a quarter turn, so the layout box has the
    // rotated size, and size it to fill the page.
    const scale = Math.min(pageH * 0.95 / r.width, width / r.height);
    const NS = 'http://www.w3.org/2000/svg';
    const outer = document.createElementNS(NS, 'svg');
    outer.setAttribute('viewBox', `0 0 ${r.height} ${r.width}`);
    outer.setAttribute('width', (r.height * scale).toFixed(1));
    outer.setAttribute('height', (r.width * scale).toFixed(1));
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('transform', `translate(0 ${r.width}) rotate(-90)`);
    svg.setAttribute('width', r.width); svg.setAttribute('height', r.height);
    svg.style.maxWidth = 'none';
    svg.replaceWith(outer); g.appendChild(svg); outer.appendChild(g);
    pre.classList.add('rotated');
  }
  // Text still containing '$' outside code is math pandoc did not parse.
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const strayDollars = [];
  while (walker.nextNode()) {
    const n = walker.currentNode;
    if (n.parentElement.closest('code, pre, .katex')) continue;
    if (n.textContent.includes('$')) strayDollars.push(n.textContent.trim().slice(0, 120));
  }
  const tooWide = [...document.querySelectorAll('body *')]
      .filter(el => el.getBoundingClientRect().right > width + 2 && !el.closest('.katex-display, pre.mmd, table, .katex-mathml'))
      .filter(el => !el.closest('pre') && !el.closest('svg') && !el.closest('[style*="zoom"]'))
      .slice(0, 5).map(el => el.tagName + ': ' + el.textContent.trim().slice(0, 60));
  return {
    katexErrors: [...document.querySelectorAll('[data-err]')].map(e => e.textContent.slice(0, 80) + '  <- ' + e.dataset.err.slice(0, 100)),
    mermaid: {total: blocks.length, errors: mermaidErrors},
    brokenImages: [...document.images].filter(im => !im.naturalWidth).map(im => im.getAttribute('src')),
    images: document.images.length,
    math: document.querySelectorAll('.katex').length,
    strayDollars, tooWide,
  };
}
"""


FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.S | re.M)
CODE_SPAN = re.compile(r"(`+)(?!`).*?(?<!`)\1(?!`)", re.S)
DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
# GitHub's inline rule: no space after the opening $ or before the closing $, no blank line inside.
INLINE = re.compile(r"(?<![\\$])\$(?=[^\s$])((?:[^$\n\\]|\\.|\n(?!\s*\n))+?)(?<=[^\s\\])\$(?![$\d])")


def protect_math(md: str):
    """Swap math for placeholder words before pandoc sees it, as GitHub does: markdown inside math
    (underscores, asterisks, backslashes) must not be interpreted. Code blocks and code spans are
    left alone. Returns the protected text and the list of (display, tex) pairs."""
    math = []

    def keep(display):
        def f(m):
            math.append((display, m.group(1)))
            return f"XMATH{len(math) - 1:05d}X"
        return f

    def outside_code(text, fn):
        parts, last = [], 0
        for m in CODE_SPAN.finditer(text):
            parts.append(fn(text[last:m.start()])); parts.append(m.group(0)); last = m.end()
        parts.append(fn(text[last:]))
        return "".join(parts)

    def prose(text):
        text = DISPLAY.sub(keep(True), text)
        return INLINE.sub(keep(False), text)

    out, last = [], 0
    for m in FENCE.finditer(md):
        out.append(outside_code(md[last:m.start()], prose)); out.append(m.group(0)); last = m.end()
    out.append(outside_code(md[last:], prose))
    return "".join(out), math


def restore_math(body: str, math) -> str:
    def f(m):
        display, tex = math[int(m.group(1))]
        cls = "math display" if display else "math inline"
        return f'<span class="{cls}">{html.escape(tex)}</span>'
    body = re.sub(r"<p>\s*XMATH(\d{5})X\s*</p>", lambda m: f(m), body)
    return re.sub(r"XMATH(\d{5})X", f, body)


def to_html(readme: Path) -> str:
    """Markdown -> standalone HTML via pandoc, with math as spans for KaTeX to render."""
    md, math = protect_math(readme.read_text())
    body = subprocess.run(["pandoc", "-f", "gfm", "-t", "html5"], input=md,
                          check=True, capture_output=True, text=True).stdout
    body = restore_math(body, math)
    # Mermaid would render class="mermaid" blocks on load by itself; RENDER_JS does it instead.
    body = re.sub(r'<pre class="mermaid"><code>(.*?)</code></pre>',
                  r'<pre class="mmd">\1</pre>', body, flags=re.S)
    title = readme.parent.name
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    base = readme.parent.as_uri() + "/"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<base href="{base}"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<script src="{MERMAID}"></script>
<style>{CSS}</style></head><body>{body}</body></html>"""


def export(page, readme: Path, out: Path) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=readme.parent, delete=False,
                                     prefix=".pdfexport_") as f:
        f.write(to_html(readme))
        tmp = Path(f.name)
    try:
        page.goto(tmp.as_uri(), wait_until="networkidle", timeout=120_000)
        page.wait_for_function("window.katex && window.mermaid", timeout=60_000)
        report = page.evaluate(RENDER_JS.replace("PAGE_H", str(PAGE_H_PX)))
        page.emulate_media(media="print")
        page.pdf(path=str(out), prefer_css_page_size=True, print_background=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:7px;width:100%;text-align:center;color:#888">'
                                 '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    finally:
        tmp.unlink()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("assignments", nargs="*", help="assignment ids (default: all with a README)")
    a = p.parse_args()
    readmes = ([ROOT / "assignments" / x / "README.md" for x in a.assignments] if a.assignments
               else sorted((ROOT / "assignments").glob("*/README.md")))
    OUT.mkdir(exist_ok=True)
    from playwright.sync_api import sync_playwright
    failed = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 1400})
        page.set_viewport_size({"width": PAGE_W_PX, "height": 1400})
        for readme in readmes:
            out = OUT / f"{readme.parent.name}.pdf"
            r = export(page, readme, out)
            pages = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True).stdout
            n_pages = re.search(r"Pages:\s+(\d+)", pages)
            problems = {k: r[k] for k in ("katexErrors", "brokenImages", "strayDollars", "tooWide") if r[k]}
            if r["mermaid"]["errors"]:
                problems["mermaid"] = r["mermaid"]["errors"]
            status = "OK " if not problems else "BAD"
            print(f"{status} {out.relative_to(ROOT)}: {n_pages.group(1) if n_pages else '?'} pages, "
                  f"{r['math']} math, {r['mermaid']['total']} diagrams, {r['images']} images", flush=True)
            if problems:
                failed.append(readme.parent.name)
                print("    " + json.dumps(problems, indent=1).replace("\n", "\n    "))
        browser.close()
    if failed:
        sys.exit(f"{len(failed)} file(s) with problems: {', '.join(failed)}")


if __name__ == "__main__":
    main()
