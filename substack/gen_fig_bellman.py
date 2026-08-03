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

def text(x, y, s, size=15, anchor="middle", fill=INK, weight="normal", family="Georgia, serif"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-family="{family}" font-weight="{weight}">{s}</text>'

def rrect(x, y, w, h, fill, stroke, sw=1.2, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def arrow_defs():
    return f'''<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M1 1L9 5L1 9Z" fill="{INK}"/>
</marker>
<marker id="arrow_red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M1 1L9 5L1 9Z" fill="{RED}"/>
</marker>
<marker id="arrow_green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
<path d="M1 1L9 5L1 9Z" fill="{GREEN}"/>
</marker>
</defs>'''

def make_bellman():
    W, H = 1040, 560
    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    parts.append(arrow_defs())

    parts.append(text(W/2, 34, "How a Q-table target actually gets built, one game", size=17, weight="bold"))

    n = 5
    box_w, box_h = 110, 62
    top_y = 80
    margin = 70
    gap = (W - 2*margin - box_w) / (n - 1)
    xs = [margin + i*gap for i in range(n)]

    labels = ["s0", "s1", "s2", "s3", "s4  (terminal)"]
    for i, x in enumerate(xs):
        is_terminal = (i == n - 1)
        fill = GREEN_FILL if is_terminal else "#FFFFFF"
        stroke = GREEN if is_terminal else GRID
        sw = 1.6 if is_terminal else 1.0
        parts.append(rrect(x, top_y, box_w, box_h, fill, stroke, sw))
        parts.append(text(x + box_w/2, top_y + 26, labels[i], size=14.5, weight="bold",
                           fill=(GREEN if is_terminal else INK)))
        if is_terminal:
            parts.append(text(x + box_w/2, top_y + 46, "win, reward = +1", size=11.5, fill=GREEN))
        else:
            parts.append(text(x + box_w/2, top_y + 46, "board state", size=11.5, fill=MUTE))

    move_y = top_y - 20
    for i in range(n - 1):
        x1 = xs[i] + box_w
        x2 = xs[i+1]
        parts.append(f'<line x1="{x1}" y1="{top_y+box_h/2 - 40}" x2="{x2}" y2="{top_y+box_h/2 - 40}" '
                      f'stroke="{INK}" stroke-width="1" marker-end="url(#arrow)"/>')
        parts.append(text((x1+x2)/2, move_y, f"move {i+1}", size=12, fill=MUTE))

    card_y = top_y + box_h + 60
    card_h = 92
    for i in range(n - 1):
        cx = (xs[i] + box_w/2 + xs[i+1] + box_w/2) / 2
        is_terminal_move = (i == n - 2)
        fill = GREEN_FILL if is_terminal_move else RED_FILL
        stroke = GREEN if is_terminal_move else RED
        cw = 200
        cxx = cx - cw/2
        parts.append(rrect(cxx, card_y, cw, card_h, fill, stroke, 1.2, rx=6))
        parts.append(text(cx, card_y + 20, f"target for move {i+1}", size=12.5, weight="bold",
                           fill=(GREEN if is_terminal_move else RED)))
        if is_terminal_move:
            parts.append(text(cx, card_y + 42, "= reward", size=13, fill=INK))
            parts.append(text(cx, card_y + 60, "(+1, known exactly,", size=11, fill=MUTE))
            parts.append(text(cx, card_y + 74, "no guessing needed)", size=11, fill=MUTE))
        else:
            parts.append(text(cx, card_y + 42, f"= 0 + γ·Q(s{i+1}, best)", size=12.5, fill=INK))
            parts.append(text(cx, card_y + 60, "(0 reward this move,", size=11, fill=MUTE))
            parts.append(text(cx, card_y + 74, "table's current guess)", size=11, fill=MUTE))

        borrow_box_x = xs[i+1] + box_w/2
        card_top = card_y
        parts.append(f'<path d="M {borrow_box_x} {top_y+box_h} '
                      f'C {borrow_box_x} {top_y+box_h+35}, {cx} {card_top-25}, {cx} {card_top}" '
                      f'fill="none" stroke="{GREEN if is_terminal_move else RED}" stroke-width="1.3" '
                      f'stroke-dasharray="{"0" if is_terminal_move else "4,3"}" '
                      f'marker-end="url(#{"arrow_green" if is_terminal_move else "arrow_red"})"/>')

    cap_y = card_y + card_h + 40
    parts.append(text(W/2, cap_y,
                       "Every earlier move's target leans on the table's current guess for the position right after it;",
                       size=13.5, fill=INK))
    parts.append(text(W/2, cap_y + 22,
                       "only the final move's target is a known fact. Replaying the game in reverse lets that guess",
                       size=13.5, fill=INK))
    parts.append(text(W/2, cap_y + 44,
                       "update within the same game, one step at a time, but each update only nudges the estimate",
                       size=13.5, fill=INK))
    parts.append(text(W/2, cap_y + 66,
                       "a small fraction of the way there, so it takes many games revisiting similar positions before",
                       size=13.5, fill=INK))
    parts.append(text(W/2, cap_y + 88,
                       "the value is trustworthy this far back, and most UTTT positions are never revisited at all.",
                       size=13.5, fill=INK))

    H2 = cap_y + 110
    parts.append("</svg>")
    svg = "\n".join(parts).replace(f'height="{H}"', f'height="{H2}"').replace(f'0 0 {W} {H}', f'0 0 {W} {H2}')
    return svg


if __name__ == "__main__":
    svg = make_bellman()
    with open("fig07_bellman_backup.svg", "w") as f:
        f.write(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to="fig07_bellman_backup.png", scale=2)
    print("done")
