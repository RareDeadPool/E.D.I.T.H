from PIL import Image
img = Image.open('scratch/now_search.png')
w, h = img.size
clusters = []
for y in range(int(h * 0.1), int(h * 0.5)):
    for x in range(int(w * 0.15), int(w * 0.75)):
        r, g, b = img.getpixel((x, y))[:3]
        if g > 165 and r < 80 and b < 130 and (g - r > 100) and (g - b > 60):
            placed = False
            for c in clusters:
                if abs(c['cx'] - x) < 30 and abs(c['cy'] - y) < 30:
                    c['pts'].append((x, y))
                    c['cx'] = sum(p[0] for p in c['pts']) // len(c['pts'])
                    c['cy'] = sum(p[1] for p in c['pts']) // len(c['pts'])
                    placed = True
                    break
            if not placed:
                clusters.append({'pts': [(x, y)], 'cx': x, 'cy': y})

for i, c in enumerate(clusters):
    if len(c['pts']) > 30:
        print(f"Cluster {i}: Center=({c['cx']}, {c['cy']}), Count={len(c['pts'])}")
