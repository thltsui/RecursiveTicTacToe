import cv2
import numpy as np

img = cv2.imread('substack/figures/fig2_send_rule.png')
if img is None:
    raise ValueError("Could not read image")

mask = np.zeros(img.shape[:2], dtype=np.uint8)

# 1. ROI 1: Blue 1
roi1 = img[290:330, 740:780]
# blue: B > 150, R < 100
blue_mask = ((roi1[:,:,0] > 150) & (roi1[:,:,2] < 100)).astype(np.uint8) * 255
mask[290:330, 740:780] = blue_mask

# 2. ROI 3: Orange 3
roi3 = img[135:185, 920:970]
# orange: B < 100, R > 200
orange_mask = ((roi3[:,:,0] < 100) & (roi3[:,:,2] > 200)).astype(np.uint8) * 255
mask[135:185, 920:970] = orange_mask

# 3. ROI 2: Text and Arrow (Dark gray)
y1, y2, x1, x2 = 240, 460, 765, 930
roi2 = img[y1:y2, x1:x2]
gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
dark_mask = (gray < 150).astype(np.uint8) * 255

kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))

lines_v = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_v)
lines_h = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_h)
grid_mask = cv2.bitwise_or(lines_v, lines_h)

text_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(grid_mask))
mask[y1:y2, x1:x2] = text_mask

# Dilate mask slightly to handle anti-aliasing edges of the text/circles
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_dilated = cv2.dilate(mask, kernel_dilate, iterations=1)

# Inpaint
result = cv2.inpaint(img, mask_dilated, 5, cv2.INPAINT_TELEA)

cv2.imwrite('substack/figures/fig2_send_rule_inpainted.png', result)
print("Saved fig2_send_rule_inpainted.png")
