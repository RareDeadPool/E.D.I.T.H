from PIL import Image

img = Image.open('scratch/believer_screen.png')
w, h = img.size
print("Size:", w, h)

# Let's inspect all green pixels in believer_screen.png
green_pts = []
for y in range(h):
    for x in range(w):
        r, g, b = img.getpixel((x, y))[:3]
        if g > 165 and r < 80 and b < 130 and (g - r > 100) and (g - b > 60):
            green_pts.append((x, y))

print(f"Total green pixels: {len(green_pts)}")
if green_pts:
    min_x = min(p[0] for p in green_pts); max_x = max(p[0] for p in green_pts)
    min_y = min(p[1] for p in green_pts); max_y = max(p[1] for p in green_pts)
    print(f"X range: [{min_x}, {max_x}], Y range: [{min_y}, {max_y}]")
    # Sample a green pixel
    print("Sample pixel color:", img.getpixel(green_pts[len(green_pts)//2]))
