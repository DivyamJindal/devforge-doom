"""Build brief/EgoMem_brief.html (rendered to PDF with any headless browser). Embeds a real generated level as SVG.

  python brief/make_brief.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "training"))
from mapgen import generate  # noqa: E402

COL = {"#": None, ".": "#e9e2d0", "P": "#2f7d4a", "E": "#d9a21b", "D": "#8b6b4a", "R": "#c0392b", "B": "#2e6db4",
       "Y": "#d4ac0d", "r": "#c0392b", "b": "#2e6db4", "y": "#d4ac0d", "m": "#7a1f1f", "h": "#e9e2d0", "a": "#e9e2d0",
       "w": "#e9e2d0", "A": "#e9e2d0"}


def level_svg(seed=3, tier=3, cell=7):
    rows = generate(seed, tier, "/tmp/egomem_brief.wad")
    ys = [y for y, r in enumerate(rows) if any(c != "#" for c in r)]
    xs = [x for x in range(len(rows[0])) if any(r[x] != "#" for r in rows)]
    y0, y1, x0, x1 = min(ys) - 1, max(ys) + 1, min(xs) - 1, max(xs) + 1
    w, h = (x1 - x0 + 1) * cell, (y1 - y0 + 1) * cell
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg"><rect width="{w}" height="{h}" fill="#1d2a30"/>']
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            c = rows[y][x]; fill = COL.get(c)
            if fill is None:
                continue
            X, Y = (x - x0) * cell, (y - y0) * cell
            if c in "rbym":
                out.append(f'<rect x="{X}" y="{Y}" width="{cell}" height="{cell}" fill="#e9e2d0"/>')
                out.append(f'<circle cx="{X + cell / 2}" cy="{Y + cell / 2}" r="{cell * 0.38}" fill="{fill}"/>')
            else:
                out.append(f'<rect x="{X}" y="{Y}" width="{cell}" height="{cell}" fill="{fill}"/>')
    out.append("</svg>")
    return "".join(out)


html = (HERE / "brief_template.html").read_text().replace("{{LEVEL_SVG}}", level_svg())
(HERE / "EgoMem_brief.html").write_text(html)
print("wrote", HERE / "EgoMem_brief.html")
