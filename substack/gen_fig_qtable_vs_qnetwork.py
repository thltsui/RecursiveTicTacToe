import cairosvg

BG = "#FBFBF8"
GRID = "#D3D1C7"
INK = "#2C2C2A"
MUTE = "#8A887E"
RED = "#B3441E"
RED_FILL = "#F3E3DB"
BLUE = "#185FA5"
BLUE_FILL = "#DCE9F5"
GREEN = "#4A7A2E"
GREEN_FILL = "#E7EFE0"
ORANGE = "#A5720E"
ORANGE_FILL = "#F3ECD9"

def text(x, y, s, size=15, anchor="middle", fill=INK, weight="normal", family="Georgia, serif"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-family="{family}" font-weight="{weight}">{s}</text>'

def rrect(x, y, w, h, fill, stroke, sw=1.2, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def arrow_defs():
    return f'''<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M1 1L9 5L1 9Z" fill="{INK}"/>
</marker>
</defs>'''

def column(cx, title, boxes, W_col):
    parts = []
    parts.append(text(cx, 40, title, size=17, weight="bold"))
    y = 70
    box_w = W_col
    box_h = 84
    xs = cx - box_w/2
    for i, (lines, fill, stroke) in enumerate(boxes):
        parts.append(rrect(xs, y, box_w, box_h, fill, stroke, 1.3))
        n_lines = len(lines)
        start_y = y + box_h/2 - (n_lines-1)*10 + 5
        for li, line in enumerate(lines):
            weight = "bold" if li == 0 else "normal"
            size = 14 if li == 0 else 12.5
            fillc = stroke if li == 0 else INK
            parts.append(text(cx, start_y + li*20, line, size=size, weight=weight, fill=fillc))
        if i < len(boxes) - 1:
            parts.append(f'<line x1="{cx}" y1="{y+box_h}" x2="{cx}" y2="{y+box_h+34}" stroke="{INK}" stroke-width="1.2" marker-end="url(#arrow)"/>')
        y += box_h + 34
    return parts, y - 34 + box_h

def make_fig():
    W = 1040
    parts = [arrow_defs()]

    left_cx = W*0.27
    right_cx = W*0.73
    col_w = 380

    left_boxes = [
        (["Q-table", "(state, action) → value,", "one row per board state seen"], RED_FILL, RED),
        (["Lookup", "exact state match required;", "unseen states: no entry, no estimate"], ORANGE_FILL, ORANGE),
    ]
    right_boxes = [
        (["Board state", "→ feature tensor"], BLUE_FILL, BLUE),
        (["Neural network", "shared weights across all states"], GREEN_FILL, GREEN),
        (["Q-value estimate", "for every action —", "generalises to unseen states"], ORANGE_FILL, ORANGE),
    ]

    left_parts, left_bottom = column(left_cx, "Tabular Q-learning", left_boxes, col_w)
    right_parts, right_bottom = column(right_cx, "Deep Q-network", right_boxes, col_w)
    parts += left_parts + right_parts

    div_y0, div_y1 = 30, max(left_bottom, right_bottom) + 20
    parts.append(f'<line x1="{W/2}" y1="{div_y0}" x2="{W/2}" y2="{div_y1}" stroke="{GRID}" stroke-width="1" stroke-dasharray="4,4"/>')

    H = max(left_bottom, right_bottom) + 40
    svg = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{W}" height="{H}" fill="{BG}"/>'] + parts + ["</svg>"]
    return "\n".join(svg)


if __name__ == "__main__":
    svg = make_fig()
    with open("fig07_qtable_vs_qnetwork.svg", "w") as f:
        f.write(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to="fig07_qtable_vs_qnetwork.png", scale=2)
    print("done")
