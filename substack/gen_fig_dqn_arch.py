import cairosvg
import textwrap

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
PURPLE = "#534AB7"
PURPLE_FILL = "#EEEDFE"
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
<marker id="arrow_purple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M1 1L9 5L1 9Z" fill="{PURPLE}"/>
</marker>
</defs>'''

def make_dqn_arch():
    W = 1080
    parts = [arrow_defs()]

    parts.append(text(W/2, 34, "DQN architecture", size=18, weight="bold"))

    boxes = [
        ("Board state\n(input)", BLUE_FILL, BLUE),
        ("Hidden layer 1\n(ReLU)", GREEN_FILL, GREEN),
        ("Hidden layer 2\n(ReLU)", GREEN_FILL, GREEN),
        ("Output layer\nQ(s, a) for\neach action", ORANGE_FILL, ORANGE),
        ("argmax\n→ move", RED_FILL, RED),
    ]
    n = len(boxes)
    box_w, box_h = 168, 92
    top_y = 70
    margin = 50
    gap = (W - 2*margin - box_w) / (n - 1)
    xs = [margin + i*gap for i in range(n)]

    for i, (label, fill, stroke) in enumerate(boxes):
        x = xs[i]
        parts.append(rrect(x, top_y, box_w, box_h, fill, stroke, 1.3))
        lines = label.split("\n")
        n_lines = len(lines)
        start_y = top_y + box_h/2 - (n_lines-1)*9 + 5
        for li, line in enumerate(lines):
            parts.append(text(x + box_w/2, start_y + li*18, line, size=13.5, weight="bold", fill=stroke))
        if i < n - 1:
            ax1 = x + box_w
            ax2 = xs[i+1]
            ay = top_y + box_h/2
            parts.append(f'<line x1="{ax1+6}" y1="{ay}" x2="{ax2-6}" y2="{ay}" stroke="{INK}" stroke-width="1.2" marker-end="url(#arrow)"/>')

    out_box_x = xs[3] + box_w/2
    arrow_top = top_y + box_h
    arrow_bot = arrow_top + 46
    parts.append(f'<line x1="{out_box_x}" y1="{arrow_top}" x2="{out_box_x}" y2="{arrow_bot}" stroke="{PURPLE}" stroke-width="1.4" marker-end="url(#arrow_purple)"/>')

    rb_y = arrow_bot + 6
    rb_w = W - 2*40
    rb_x = 40
    caption = ("(state, action, reward, next_state) transitions are stored here and sampled "
               "in random mini-batches, which breaks the correlation between consecutive experiences.")
    wrapped = textwrap.wrap(caption, width=78)
    line_h = 22
    pad_top = 34
    pad_bot = 18
    rb_h = pad_top + len(wrapped) * line_h + pad_bot
    parts.append(rrect(rb_x, rb_y, rb_w, rb_h, PURPLE_FILL, PURPLE, 1.3))
    parts.append(text(rb_x + rb_w/2, rb_y + 24, "Replay buffer", size=14.5, weight="bold", fill=PURPLE))
    for li, line in enumerate(wrapped):
        parts.append(text(rb_x + rb_w/2, rb_y + pad_top + 12 + li*line_h, line, size=13, fill=INK))

    H = rb_y + rb_h + 30
    svg = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{W}" height="{H}" fill="{BG}"/>'] + parts + ["</svg>"]
    return "\n".join(svg)


if __name__ == "__main__":
    svg = make_dqn_arch()
    with open("fig07_dqn_architecture.svg", "w") as f:
        f.write(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to="fig07_dqn_architecture.png", scale=2)
    print("done")
