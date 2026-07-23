import numpy as np
from PIL import Image

img_pil = Image.open('substack/figures/fig2_send_rule.png').convert('RGB')
img = np.array(img_pil)

# 1. Colors
yellow_bg = img[100, 900].copy()
white_bg = img[100, 400].copy()
thick_line_color = img[315, 400].copy() # thick horizontal line
thin_line_color = img[150, 400].copy()  # thin horizontal line
orange_color = img[160, 940].copy() # center of the 3

print("yellow_bg:", yellow_bg)
print("white_bg:", white_bg)
print("thick_line_color:", thick_line_color)
print("thin_line_color:", thin_line_color)

# We will overwrite specific bounding boxes.

# Bounding box for ① (above X)
# x: 740 to 770, y: 300 to 325
img[290:330, 740:780] = white_bg

# Bounding box for ② and text and arrow
# The arrow starts near X and goes to top-right. 
# Let's erase a large block from x=765 to 920, y=240 to 455
img[240:460, 765:930] = white_bg

# Bounding box for ③ (in top-right)
# x: 920 to 970, y: 140 to 180
img[135:185, 920:970] = yellow_bg

# Now we need to redraw the grid lines that were erased.
# Thick vertical line at x ~ 809? Let's search for the exact x.
thick_x = None
for x in range(800, 820):
    if np.array_equal(img[100, x], thick_line_color):
        thick_x = x
        break

# Thick horizontal line at y ~ 315.
thick_y = None
for y in range(310, 325):
    if np.array_equal(img[y, 400], thick_line_color):
        thick_y = y
        break

# Thin horizontal lines near y=396 and y=151
thin_y1 = None
for y in range(390, 405):
    if np.array_equal(img[y, 400], thin_line_color):
        thin_y1 = y
        break

thin_y2 = None
for y in range(145, 160):
    if np.array_equal(img[y, 400], thin_line_color):
        thin_y2 = y
        break

# Thin vertical line near x=888
thin_x1 = None
for x in range(880, 895):
    if np.array_equal(img[100, x], thin_line_color):
        thin_x1 = x
        break

print(f"Thick lines: x={thick_x}, y={thick_y}")
print(f"Thin lines: x1={thin_x1}, y1={thin_y1}, y2={thin_y2}")

# Redraw thick vertical line
if thick_x:
    img[240:460, thick_x-1:thick_x+3] = thick_line_color

# Redraw thick horizontal line
if thick_y:
    img[thick_y-1:thick_y+3, 740:930] = thick_line_color

# Redraw thin vertical line
if thin_x1:
    img[240:460, thin_x1:thin_x1+1] = thin_line_color

# Redraw thin horizontal line at y1
if thin_y1:
    img[thin_y1:thin_y1+1, 765:930] = thin_line_color

# Redraw thin horizontal line at y2
if thin_y2:
    img[thin_y2:thin_y2+1, 920:970] = thin_line_color

# Color yellow for the top part of the erased region right of thick_x
if thick_y:
    for y in range(240, thick_y):
        for x in range(765, 930):
            # Only color yellow if it's right of thick_x
            if thick_x and x > thick_x:
                if not np.array_equal(img[y, x], thick_line_color) and not np.array_equal(img[y, x], thin_line_color):
                    img[y, x] = yellow_bg

# Save output
Image.fromarray(img).save('substack/figures/fig2_send_rule.png')
print("Saved fig2_send_rule.png")
