from PIL import Image

img = Image.open("spotify_track_uri.bmp")
w, h = img.size
print(f"Dimensions: {w}x{h}")

# Crop bottom 150 pixels
bottom = img.crop((0, h - 120, w, h))
bottom.save("spotify_bottom.png")
print("Saved spotify_bottom.png")
