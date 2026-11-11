"""Generate the ManyPred logo and icon SVGs (light/dark), text converted to outlines.

Same family as ManySeg's logo (Lato Bold, slate palette, teal and indigo). The mark is a
predicted mask with two overlapping patch frames: tiles that are blended into one prediction.

Usage: uv run --no-project --with fonttools python scripts/make_logo.py docs/assets
"""

import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(sys.argv[1])
FONT = TTFont("/usr/share/fonts/truetype/lato/Lato-Bold.ttf")
GS = FONT.getGlyphSet()
CMAP = FONT.getBestCmap()
UPM = FONT["head"].unitsPerEm
HMTX = FONT["hmtx"]


def text_path(s, x, baseline, size, tracking=0.0):
    scale = size / UPM
    pen = SVGPathPen(GS)
    for ch in s:
        g = CMAP[ord(ch)]
        GS[g].draw(TransformPen(pen, (scale, 0, 0, -scale, x, baseline)))
        x += HMTX[g][0] * scale + tracking * size
    return pen.getCommands(), x


# 4x4 predicted mask: 0 = background, 1 = class A, 2 = class B
MASK = [
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 2, 2],
    [0, 2, 2, 2],
]

THEMES = {
    "light": dict(
        ink="#0F172A",
        bg="#E2E8F0",
        a="#0D9488",
        b="#6366F1",
        word="#0D9488",
        plate="#F8FAFC",
        edge="#CBD5E1",
        frame_a="#334155",
        frame_b="#334155",
    ),
    "dark": dict(
        ink="#F1F5F9",
        bg="#334155",
        a="#2DD4BF",
        b="#818CF8",
        word="#2DD4BF",
        plate="#1E293B",
        edge="#475569",
        frame_a="#CBD5E1",
        frame_b="#CBD5E1",
    ),
}

CELL, GAP, R = 22, 4, 5
INNER = 4 * CELL + 3 * GAP  # 100
PP = 6  # plate padding around the cells
GRID = INNER + 2 * PP
PAD = 12  # room for the frames, which stick out of the plate


def mark(t, ox, oy):
    parts = [
        f'<rect x="{ox + 0.75}" y="{oy + 0.75}" width="{GRID - 1.5}" height="{GRID - 1.5}" '
        f'rx="{R + 3}" fill="{t["plate"]}" stroke="{t["edge"]}" stroke-width="1.5"/>'
    ]
    cx, cy = ox + PP, oy + PP
    fill = {0: t["bg"], 1: t["a"], 2: t["b"]}
    for r, row in enumerate(MASK):
        for c, v in enumerate(row):
            parts.append(
                f'<rect x="{cx + c * (CELL + GAP)}" y="{cy + r * (CELL + GAP)}" width="{CELL}" '
                f'height="{CELL}" rx="{R}" fill="{fill[v]}"/>'
            )
    # two overlapping patch frames: top-left 3x3 cells and bottom-right 3x3 cells
    span = 3 * CELL + 2 * GAP + 8
    for (r0, c0), col in (((0, 0), t["frame_a"]), ((1, 1), t["frame_b"])):
        x = cx + c0 * (CELL + GAP) - 4
        y = cy + r0 * (CELL + GAP) - 4
        parts.append(
            f'<rect x="{x}" y="{y}" width="{span}" height="{span}" rx="{R + 3}" fill="none" '
            f'stroke="{col}" stroke-width="3" stroke-dasharray="7 4" stroke-linecap="round"/>'
        )
    return "\n  ".join(parts)


def build(theme, with_text=True):
    t = THEMES[theme]
    ox, oy = PAD, PAD
    m = mark(t, ox, oy)
    mark_w = ox + GRID + PAD
    h = oy + GRID + PAD
    if not with_text:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {mark_w} {h}" width="{mark_w}" '
            f'height="{h}" role="img" aria-label="ManyPred">\n  {m}\n</svg>\n'
        )
    h += 6  # room for the descenders
    size = 72
    baseline = oy + GRID / 2 + size * 0.34
    tx = mark_w + 20
    many, x = text_path("Many", tx, baseline, size, tracking=-0.01)
    pred, x = text_path("Pred", x, baseline, size, tracking=-0.01)
    w = int(x + PAD + 4)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="ManyPred">\n  {m}\n  <path fill="{t["ink"]}" d="{many}"/>\n'
        f'  <path fill="{t["word"]}" d="{pred}"/>\n</svg>\n'
    )


OUT.mkdir(parents=True, exist_ok=True)
for theme in THEMES:
    (OUT / f"logo-{theme}.svg").write_text(build(theme))
    (OUT / f"icon-{theme}.svg").write_text(build(theme, with_text=False))
print("ok")
