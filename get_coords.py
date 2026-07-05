import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('substack/figures/fig2_send_rule.png')
plt.figure(figsize=(12, 9))
plt.imshow(img)
plt.grid(True, color='red', linestyle='-', linewidth=0.5)
plt.xticks(range(0, img.width, 50), rotation=90)
plt.yticks(range(0, img.height, 50))
plt.savefig('temp_grid.png', dpi=150)
