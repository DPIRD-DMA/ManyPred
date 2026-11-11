"""Generate the README pipeline diagram (light/dark SVGs, text converted to outlines).

Same style as ManySeg's architecture diagram (Lato, slate palette, teal/indigo/amber).

Usage: uv run --no-project --with fonttools python scripts/make_diagram.py docs/assets
"""

import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(sys.argv[1])
FONTS = {}
for weight, file in (("regular", "Lato-Regular.ttf"), ("bold", "Lato-Bold.ttf")):
    f = TTFont(f"/usr/share/fonts/truetype/lato/{file}")
    FONTS[weight] = (f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm, f["hmtx"])


def text_width(s, size, weight="regular", tracking=0.0):
    _, cmap, upm, hmtx = FONTS[weight]
    return sum(hmtx[cmap[ord(c)]][0] * size / upm + tracking * size for c in s)


def text(s, x, y, size, fill, weight="regular", anchor="start", tracking=0.0):
    gs, cmap, upm, hmtx = FONTS[weight]
    if anchor == "middle":
        x -= text_width(s, size, weight, tracking) / 2
    elif anchor == "end":
        x -= text_width(s, size, weight, tracking)
    scale = size / upm
    pen = SVGPathPen(gs)
    for c in s:
        g = cmap[ord(c)]
        gs[g].draw(TransformPen(pen, (scale, 0, 0, -scale, x, y)))
        x += hmtx[g][0] * scale + tracking * size
    return f'<path fill="{fill}" d="{pen.getCommands()}"/>'


THEMES = {
    "light": dict(
        ink="#0F172A",
        muted="#64748B",
        line="#94A3B8",
        plate="#F8FAFC",
        edge="#CBD5E1",
        soft="#E2E8F0",
        a="#0D9488",
        a_soft="#CCFBF1",
        b="#6366F1",
        b_soft="#E0E7FF",
        c="#F59E0B",
        c_soft="#FEF3C7",
        frame="#334155",
        bg="#FFFFFF",
    ),
    "dark": dict(
        ink="#F1F5F9",
        muted="#CBD5E1",
        line="#7C8BA1",
        plate="#1E293B",
        edge="#475569",
        soft="#334155",
        a="#2DD4BF",
        a_soft="#134E4A",
        b="#818CF8",
        b_soft="#312E81",
        c="#FBBF24",
        c_soft="#78350F",
        frame="#CBD5E1",
        bg="#0D1117",
    ),
}

W, H = 1000, 410
COLS = {"in": 95, "tile": 290, "gpu": 495, "blend": 690, "out": 885}
ROW = 125  # centre line of the icons
LABEL = 222  # titles under the icons
RULE_Y = 282


def build(theme):
    t = THEMES[theme]
    el = []
    add = el.append

    def arrow(x0, y0, x1, y1, col=None):
        add(
            f'<path d="M{x0} {y0} L{x1} {y1}" stroke="{col or t["line"]}" stroke-width="1.5" '
            f'fill="none" marker-end="url(#arrow)"/>'
        )

    def label(key, y, title, sub):
        add(text(title, COLS[key], y, 16.5, t["ink"], "bold", "middle"))
        add(text(sub, COLS[key], y + 18, 14, t["muted"], anchor="middle"))

    for key, name in (
        ("in", "INPUTS"),
        ("tile", "TILING"),
        ("gpu", "ON THE GPU"),
        ("blend", "BLENDING"),
        ("out", "OUTPUTS"),
    ):
        add(text(name, COLS[key], 34, 13.5, t["muted"], "bold", "middle", tracking=0.12))

    # ---- inputs: a band stack (the scene) and files read by load_func
    cx = COLS["in"]
    for i in reversed(range(4)):
        size, off = 70, 7
        x = cx - size / 2 - 1.5 * off + i * off
        y = ROW - size / 2 + 1.5 * off - i * off
        fill = t["a"] if i == 0 else t["a_soft"]
        add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" rx="7" fill="{fill}" '
            f'stroke="{t["bg"]}" stroke-width="1.5"/>'
        )
    label("in", LABEL, "scene", "any bands · any size")

    # ---- tiling: the scene with overlapping patches; the all-nodata corner patch is skipped
    tx, size = COLS["tile"], 128
    x0, y0 = tx - size / 2, ROW - size / 2
    add(
        f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" rx="8" fill="{t["a_soft"]}" '
        f'stroke="{t["edge"]}" stroke-width="1.5"/>'
    )
    add(
        f'<path d="M{x0 + 8} {y0} L{x0 + 56} {y0} L{x0 + 56} {y0 + 56} L{x0} {y0 + 56} L{x0} {y0 + 8} '
        f'Q{x0} {y0}, {x0 + 8} {y0} Z" fill="{t["soft"]}"/>'
    )
    patch, stride = 56, 36
    for r in range(3):
        for c in range(3):
            px, py = x0 + c * stride, y0 + r * stride
            if r == 0 and c == 0:  # all nodata: skipped
                add(
                    f'<path d="M{px + 14} {py + 14} l18 18 M{px + 32} {py + 14} l-18 18" '
                    f'stroke="{t["line"]}" stroke-width="2" stroke-linecap="round"/>'
                )
                continue
            hot = (r, c) == (1, 1)
            dash = "" if hot else 'stroke-dasharray="4 3"'
            add(
                f'<rect x="{px:.1f}" y="{py:.1f}" width="{patch}" height="{patch}" rx="5" '
                f'fill="none" stroke="{t["b"] if hot else t["frame"]}" '
                f'stroke-width="{2.5 if hot else 1.2}" stroke-opacity="{1 if hot else 0.55}" '
                f"{dash}/>"
            )
    label("tile", LABEL, "patches", "overlap · nodata skipped")
    arrow(COLS["in"] + 52, ROW, x0 - 10, ROW)

    # ---- on the GPU: a batch of patches -> normalise -> model(s) -> activation
    gx = COLS["gpu"]
    for i in reversed(range(4)):  # a batch
        bx, by = gx - 82 + i * 6, ROW - 17 - i * 6 + 9
        add(
            f'<rect x="{bx}" y="{by}" width="34" height="34" rx="5" fill="{t["b_soft"]}" '
            f'stroke="{t["b"]}" stroke-width="1.2"/>'
        )
    add(text("batch", gx - 63, ROW + 40, 14, t["muted"], anchor="middle"))
    arrow(x0 + size + 8, ROW, gx - 92, ROW)
    steps = [
        ("normalise", t["soft"], t["ink"]),
        ("model(s)", t["ink"], t["bg"]),
        ("activation", t["soft"], t["ink"]),
    ]
    for i, (name, fill, ink) in enumerate(steps):
        sy = ROW - 47 + i * 34
        add(f'<rect x="{gx - 22}" y="{sy}" width="92" height="26" rx="13" fill="{fill}"/>')
        add(text(name, gx + 24, sy + 18, 13.5, ink, "bold", "middle"))
    arrow(gx - 40, ROW, gx - 28, ROW)
    label("gpu", LABEL, "batched inference", "copies overlap compute")

    # ---- blending: gradient weights, weighted sum / weight sum
    bx = COLS["blend"]
    for i, op in enumerate((0.18, 0.35, 0.55, 0.78, 1.0)):
        s = 92 - i * 16
        add(
            f'<rect x="{bx - s / 2}" y="{ROW - s / 2}" width="{s}" height="{s}" '
            f'rx="{8 - i}" fill="{t["c"]}" fill-opacity="{op}"/>'
        )
    arrow(gx + 76, ROW, bx - 56, ROW)
    label("blend", LABEL, "Σ w·p / Σ w", "weights fade at the edges")

    # ---- outputs: a mosaic, a reduced-resolution head, an image-level value
    ox = COLS["out"]
    mask = [[1, 1, 0, 0, 0], [1, 1, 1, 0, 2], [0, 1, 1, 2, 2], [0, 0, 2, 2, 2], [0, 2, 2, 2, 0]]
    fill = {0: t["soft"], 1: t["a"], 2: t["b"]}
    cell, gap = 11, 2.5
    gw = 5 * cell + 4 * gap
    mx, my = ox - gw / 2 - 26, ROW - gw / 2
    add(
        f'<rect x="{mx - 5}" y="{my - 5}" width="{gw + 10}" height="{gw + 10}" rx="8" '
        f'fill="{t["plate"]}" stroke="{t["edge"]}" stroke-width="1.5"/>'
    )
    for r, row in enumerate(mask):
        for c, v in enumerate(row):
            add(
                f'<rect x="{mx + c * (cell + gap)}" y="{my + r * (cell + gap)}" width="{cell}" '
                f'height="{cell}" rx="3" fill="{fill[v]}"/>'
            )
    qx, qy, qc = ox + 30, ROW - 32, 13  # 1/4-resolution head: coarser cells
    for r in range(2):
        for c in range(2):
            add(
                f'<rect x="{qx + c * (qc + 3)}" y="{qy + r * (qc + 3)}" width="{qc}" height="{qc}" '
                f'rx="3" fill="{t["c"]}" fill-opacity="{0.45 + 0.18 * (r + c)}"/>'
            )
    add(
        f'<rect x="{ox + 22}" y="{ROW + 14}" width="50" height="26" rx="13" fill="{t["plate"]}" '
        f'stroke="{t["edge"]}" stroke-width="1.5"/>'
    )
    add(text("0.82", ox + 47, ROW + 32, 14, t["ink"], "bold", "middle"))
    arrow(bx + 54, ROW, mx - 14, ROW)
    label("out", LABEL, "predictions", "arrays · GeoTIFFs")

    # ---- what each column takes
    opts = {
        "in": ["arrays · tensors · dicts", "files via load_func", "lazy readers"],
        "tile": [
            "any patch size and overlap",
            "shifted off nodata edges",
            "inputs at other resolutions",
        ],
        "gpu": ["any nn.Module · ensembles", "fp16 · bf16 · compile", "CUDA · MPS · CPU"],
        "blend": ["float32 · fp16 · bf16", "on each output's grid", "GPU or CPU mosaic"],
        "out": [
            "logits · probs · argmax",
            "reduced-resolution heads",
            "image heads per patch",
            "written in row blocks",
        ],
    }
    add(
        f'<line x1="24" y1="{RULE_Y}" x2="{W - 24}" y2="{RULE_Y}" stroke="{t["edge"]}" '
        f'stroke-width="1"/>'
    )
    for key, lines in opts.items():
        for i, s in enumerate(lines):
            add(text(s, COLS[key], RULE_Y + 27 + i * 20, 14, t["muted"], anchor="middle"))

    defs = (
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 1 L9 5 L0 9 z" '
        f'fill="{t["line"]}"/></marker></defs>'
    )
    body = "\n  ".join(el)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="ManyPred pipeline: inputs, tiling, batched '
        f'inference on the GPU, blending, outputs">\n  {defs}\n  {body}\n</svg>\n'
    )


OUT.mkdir(parents=True, exist_ok=True)
for theme in THEMES:
    (OUT / f"pipeline-{theme}.svg").write_text(build(theme))
print("ok")
