"""Build an HTML page next to every README.md in the project (stdlib only).

    python3 tools/build_docs.py

Supports the Markdown used in this project: headings, paragraphs, bullet and
numbered lists, tables, fenced code blocks, inline code, bold and links
(links to .md files are rewritten to the generated .html pages).

Two-column "server | client" blocks (HTML comments, invisible on GitHub):

    <!-- sides -->
    <!-- server -->
    **Optional column title**
    ...markdown for server_side...
    <!-- client -->
    ...markdown for the PC...
    <!-- /sides -->
Edit the README.md files, then rerun this script; never edit the .html.
"""

from __future__ import annotations

import base64
import html
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNER = ROOT / "assets" / "lan_mesh.svg"  # regenerate: python3 tools/make_mesh_illustration.py
BANNER_ALT = ("Illustration of a local network mesh: server devices A to D and a local client "
              "exchange packets in both directions through a Wi-Fi switch, among other devices on the LAN")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist"}

_INLINE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<href>[^)\s]+)\)|`(?P<code>[^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_UL = re.compile(r"^\s*[-*]\s+(.*)")
_OL = re.compile(r"^\s*(\d+)\.\s+(.*)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)")


# --------------------------------------------------------------------------- inline
def inline(text: str) -> str:
    out, pos = [], 0
    for m in _INLINE.finditer(text):
        out.append(_plain(text[pos : m.start()]))
        if m["code"] is not None:
            out.append(f"<code>{html.escape(m['code'])}</code>")
        else:
            href = re.sub(r"(^|/)README\.md(#|$)", r"\1README.html\2", m["href"])
            out.append(f'<a href="{html.escape(href, quote=True)}">{inline(m["text"])}</a>')
        pos = m.end()
    out.append(_plain(text[pos:]))
    return "".join(out)


def _plain(text: str) -> str:
    return _BOLD.sub(r"<strong>\1</strong>", html.escape(text, quote=False))


def slugify(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-") or "section"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


# --------------------------------------------------------------------------- blocks
SIDES_START, SIDES_END = "<!-- sides -->", "<!-- /sides -->"
SIDE_MARKERS = {"<!-- server -->": "server", "<!-- client -->": "client"}
SIDE_DEFAULT_LABEL = {"server": "Server side", "client": "Client side"}
_COL_MARKER = re.compile(r"^<!--\s*col\s+([\w.+-]+)\s*-->$")
_COMMENT = re.compile(r"^\s*<!--.*-->\s*$")
_DETAILS_OPEN = re.compile(r"^\s*<details>\s*(?:<summary>(.*?)</summary>)?\s*$")
_BOLD_LINE = re.compile(r"^\s*\*\*(.+?)\*\*:?\s*$")


@dataclass
class _State:
    title: str = "README"
    toc: list[tuple[int, str, str]] = field(default_factory=list)
    used: set[str] = field(default_factory=set)


def render(markdown: str) -> tuple[str, str, list[tuple[int, str, str]]]:
    """Return (title, body_html, toc[(level, id, text)])."""
    state = _State()
    body = _blocks(markdown.splitlines(), state)
    return state.title, body, state.toc


def _sides(lines: list[str], state: _State) -> str:
    """Side-by-side columns, in the order they appear.

    <!-- server --> / <!-- client --> give the SERVER / CLIENT styles; any
    <!-- col NAME --> gives a neutral column with a NAME badge (e.g. ufw,
    firewalld). Each column may start with a bold line used as its title.
    """
    parts: list[tuple[str, str, list[str]]] = []  # (style, badge, lines)
    for line in lines:
        marker = line.strip()
        if (side := SIDE_MARKERS.get(marker)) is not None:
            parts.append((side, side.upper(), []))
        elif m := _COL_MARKER.match(marker):
            parts.append(("generic", m[1].upper(), []))
        elif parts:
            parts[-1][2].append(line)
    columns = []
    for style, badge, body in parts:
        while body and not body[0].strip():
            body.pop(0)
        label = SIDE_DEFAULT_LABEL.get(style, "")
        if body and (m := _BOLD_LINE.match(body[0])):
            label, body = m[1], body[1:]
        columns.append(
            f'<section class="side side-{style}"><div class="side-head"><span class="badge">{html.escape(badge)}</span>'
            f"<span>{inline(label)}</span></div>{_blocks(body, state) if body else '<p>Nothing to do here.</p>'}</section>"
        )
    return f'<div class="sides" style="--cols:{max(1, len(columns))}">{"".join(columns)}</div>'


def _blocks(lines: list[str], state: _State) -> str:
    out: list[str] = []
    toc, used = state.toc, state.used
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        if line.strip() == SIDES_START:
            i += 1
            block = []
            while i < len(lines) and lines[i].strip() != SIDES_END:
                block.append(lines[i])
                i += 1
            i += 1
            out.append(_sides(block, state))
            continue

        if (m := _DETAILS_OPEN.match(line)) is not None:
            summary, depth, inner = m[1] or "More", 1, []
            i += 1
            while i < len(lines):
                if _DETAILS_OPEN.match(lines[i]):
                    depth += 1
                elif lines[i].strip() == "</details>":
                    depth -= 1
                    if depth == 0:
                        break
                inner.append(lines[i])
                i += 1
            i += 1  # skip </details>
            out.append(f'<details class="deep"><summary>{inline(summary)}</summary>'
                       f'<div class="deep-body">{_blocks(inner, state)}</div></details>')
            continue

        if _COMMENT.match(line):
            i += 1
            continue

        if line.startswith("```"):
            lang = line[3:].strip()
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # closing fence
            label = f'<span class="lang">{html.escape(lang)}</span>' if lang else ""
            out.append(
                f'<div class="code lang-{html.escape(lang or "text")}">{label}<button class="copy" type="button" aria-label="Copy code">Copy</button>'
                f"<pre><code>{html.escape(chr(10).join(code))}</code></pre></div>"
            )
            continue

        if m := _HEADING.match(line):
            level, text = len(m[1]), inline(m[2].strip())
            if level == 1:
                state.title = re.sub(r"<[^>]+>", "", text)
                out.append(f"<h1>{text}</h1>")
            else:
                anchor = slugify(text, used)
                toc.append((level, anchor, text))
                out.append(f'<h{level} id="{anchor}"><a class="anchor" href="#{anchor}">{text}</a></h{level}>')
            i += 1
            continue

        if line.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = _cells(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            thead = "".join(f"<th>{inline(c)}</th>" for c in header)
            tbody = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f'<div class="table"><table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>')
            continue

        if _UL.match(line):
            items = []
            while i < len(lines) and (m := _UL.match(lines[i])):
                items.append(f"<li>{inline(m[1])}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        if m := _OL.match(line):
            start = int(m[1])
            items = []
            while i < len(lines) and (m := _OL.match(lines[i])):
                items.append(f"<li>{inline(m[2])}</li>")
                i += 1
            attr = f' start="{start}"' if start != 1 else ""
            out.append(f"<ol{attr}>{''.join(items)}</ol>")
            continue

        para = []
        while i < len(lines) and lines[i].strip() and not _is_block_start(lines, i):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")

    return "\n".join(out)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _is_block_start(lines: list[str], i: int) -> bool:
    line = lines[i]
    return bool(
        line.startswith("```") or _HEADING.match(line) or _UL.match(line) or _OL.match(line)
        or line.lstrip().startswith("|") or _COMMENT.match(line)
        or _DETAILS_OPEN.match(line) or line.strip() == "</details>"
    )


# --------------------------------------------------------------------------- page
CSS = """
:root{--bg:#fbfbfa;--surface:#fff;--text:#1d1f23;--muted:#5d636e;--border:#e3e4e8;--accent:#2756c7;
--code-bg:#f3f4f6;--pre-bg:#16181d;--pre-text:#e6e8ec;--th:#f5f6f8;--server:#b45309;--client:#2756c7;--deep:#0e7490;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121317;--surface:#1a1c21;--text:#e7e8eb;
--muted:#a0a5ae;--border:#2c2f36;--accent:#8bb0ff;--code-bg:#262931;--pre-bg:#0c0d10;--pre-text:#e6e8ec;--th:#21242a;--server:#f5a524;--client:#8bb0ff;--deep:#67e8f9;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121317;--surface:#1a1c21;--text:#e7e8eb;--muted:#a0a5ae;--border:#2c2f36;--accent:#8bb0ff;
--code-bg:#262931;--pre-bg:#0c0d10;--pre-text:#e6e8ec;--th:#21242a;--server:#f5a524;--client:#8bb0ff;--deep:#67e8f9;color-scheme:dark}
*{box-sizing:border-box}
html{scroll-padding-top:72px}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
header.top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);
border-bottom:1px solid var(--border)}
header.top .inner{max-width:1160px;margin:0 auto;padding:10px 16px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
header.top .brand{font-weight:650;margin-right:12px}
header.top nav a{padding:5px 10px;border-radius:7px;text-decoration:none;color:var(--muted);font-size:14px}
header.top nav a[aria-current="page"]{background:var(--code-bg);color:var(--text);font-weight:600}
.layout{max-width:1160px;margin:0 auto;padding:24px 16px 64px;display:grid;grid-template-columns:minmax(0,1fr);gap:40px}
@media (min-width:980px){.layout{grid-template-columns:220px minmax(0,1fr)}}
aside.toc{display:none}
@media (min-width:980px){aside.toc{display:block;position:sticky;top:76px;align-self:start;max-height:calc(100vh - 96px);overflow:auto}}
aside.toc p{margin:0 0 8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
aside.toc a{display:block;padding:3px 0;color:var(--muted);text-decoration:none;font-size:14px}
aside.toc a.l3{padding-left:14px;font-size:13px}
aside.toc a:hover{color:var(--accent)}
main{min-width:0;max-width:820px}
h1{font-size:2.1rem;line-height:1.2;margin:.2em 0 .6em;letter-spacing:-.01em}
h2{font-size:1.45rem;margin:2.2em 0 .6em;padding-bottom:.3em;border-bottom:1px solid var(--border)}
h3{font-size:1.15rem;margin:1.8em 0 .5em}
h2 a.anchor,h3 a.anchor{color:inherit;text-decoration:none}
h2 a.anchor:hover::after,h3 a.anchor:hover::after{content:" #";color:var(--muted)}
p,ul,ol{margin:0 0 1em}
li{margin:.25em 0}
code{font:0.88em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code-bg);padding:.12em .38em;border-radius:5px;
overflow-wrap:anywhere}
.code{position:relative;margin:0 0 1.2em;background:var(--pre-bg);border-radius:10px;border:1px solid var(--border)}
.code pre{margin:0;padding:16px;overflow-x:auto;color:var(--pre-text);font-size:13.5px;line-height:1.55}
.code pre code{background:none;padding:0;color:inherit;font-size:inherit;overflow-wrap:normal}
.code .lang{position:absolute;top:8px;right:66px;font:11px ui-monospace,monospace;color:#8b919c;text-transform:uppercase}
.code .copy{position:absolute;top:6px;right:8px;font:12px system-ui,sans-serif;color:#c9cdd4;background:#2a2d35;border:1px solid #3a3e47;
border-radius:6px;padding:3px 9px;cursor:pointer;opacity:.85}
.code .copy:hover,.code .copy:focus-visible{opacity:1}
.code:has(.copy) pre{padding-top:34px}
.table{overflow-x:auto;margin:0 0 1.2em;border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:15px;background:var(--surface)}
th,td{padding:9px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--border)}
th{background:var(--th);font-weight:600}
tbody tr:last-child td{border-bottom:0}
.sides{display:grid;grid-template-columns:minmax(0,1fr);gap:16px;margin:0 0 1.4em}
@media (min-width:760px){.sides{grid-template-columns:repeat(var(--cols,2),minmax(0,1fr))}}
.side{min-width:0;background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--side);border-radius:10px;padding:0 14px 4px}
.side-server{--side:var(--server)}.side-client{--side:var(--client)}.side-generic{--side:var(--muted)}
.side-head{display:flex;align-items:center;gap:10px;margin:0 -14px 12px;padding:10px 14px;border-bottom:1px solid var(--border);font-weight:600}
.side .badge{font:700 11px/1 ui-monospace,monospace;letter-spacing:.06em;color:var(--side);border:1px solid var(--side);border-radius:5px;padding:4px 6px}
.side ol,.side ul{padding-left:1.3em}.side .code pre{font-size:12px;padding-left:12px;padding-right:12px}.side .code.lang-bash pre{white-space:pre-wrap;overflow-wrap:anywhere}
.banner{display:block;width:100%;height:auto;aspect-ratio:16/9;margin:0 0 28px;border-radius:14px;border:1px solid var(--border);background:#0a1020;box-shadow:0 10px 30px -18px rgba(10,16,32,.6)}
details.deep{margin:-.55em 0 1em 1.5em;padding:2px 0 2px 12px;border-left:2px solid var(--border)}
details.deep[open]{border-left-color:color-mix(in srgb,var(--accent) 45%,var(--border))}
details.deep>summary{cursor:pointer;list-style:none;font-size:12.5px;font-weight:600;color:var(--muted);letter-spacing:.02em;padding:2px 0;width:fit-content}
details.deep>summary::-webkit-details-marker{display:none}
details.deep>summary::before{content:"▸";display:inline-block;width:1.1em;transition:transform .15s}
details.deep[open]>summary::before{transform:rotate(90deg)}
details.deep>summary:hover,details.deep>summary:focus-visible{color:var(--accent)}
.deep-body{font-size:13.5px;line-height:1.55;font-style:italic;color:var(--deep);padding:4px 0 2px}
.deep-body p,.deep-body ul,.deep-body ol{margin:.45em 0}.deep-body li{margin:.15em 0}
.deep-body code{font-style:normal}.deep-body strong{color:inherit}
.deep-body .code{font-style:normal;margin:.5em 0}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--border);color:var(--muted);font-size:13px}
"""

JS = """
document.querySelectorAll('.code .copy').forEach(btn=>{
  btn.addEventListener('click',async()=>{
    const text=btn.parentElement.querySelector('code').innerText;
    try{await navigator.clipboard.writeText(text);btn.textContent='Copied';}
    catch(e){btn.textContent='Select & copy';}
    setTimeout(()=>btn.textContent='Copy',1500);
  });
});
"""


def page(title: str, body: str, toc: list[tuple[int, str, str]], nav: str, banner: str = "") -> str:
    toc_html = "".join(f'<a class="l{lvl}" href="#{a}">{t}</a>' for lvl, a, t in toc if lvl <= 3)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} docs</title>
<style>{CSS}</style>
</head>
<body>
<header class="top"><div class="inner"><span class="brand">udp_link project</span><nav>{nav}</nav></div></header>
<div class="layout">
<aside class="toc" aria-label="On this page"><p>On this page</p>{toc_html}</aside>
<main>
{banner}{body}
<footer>Generated from README.md by <code>tools/build_docs.py</code>. Edit the Markdown, then rerun the script.</footer>
</main>
</div>
<script>{JS}</script>
</body>
</html>
"""


def find_readmes() -> list[Path]:
    found = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        if "README.md" in files:
            found.append(Path(folder) / "README.md")
    return found


def main() -> None:
    readmes = find_readmes()
    docs = [(p, render(p.read_text(encoding="utf-8"))) for p in readmes]
    for path, (title, body, toc) in docs:
        target = path.with_suffix(".html")
        nav = "".join(
            f'<a href="{html.escape(os.path.relpath(other.with_suffix(".html"), target.parent))}"'
            f'{" aria-current=" + chr(34) + "page" + chr(34) if other == path else ""}>{html.escape(t)}</a>'
            for other, (t, _, _) in docs
        )
        banner = ""
        if BANNER.exists():
            # embedded, so the page works anywhere (email, file share, no assets/ folder)
            src = "data:image/svg+xml;base64," + base64.b64encode(BANNER.read_bytes()).decode("ascii")
            banner = (f'<img class="banner" src="{src}" width="1600" height="900" '
                      f'alt="{html.escape(BANNER_ALT, quote=True)}">\n')
        target.write_text(page(title, body, toc, nav, banner), encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
