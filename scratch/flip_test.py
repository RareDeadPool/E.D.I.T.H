from PIL import Image

# Remember PrintWindow gives bottom-up DIB or top-down?
# In our previous view_file of scratch/current_spotify.png, the image was upside down because DIB was read bottom-up!
# Let's inspect the actual orientation and find where the search bar is.

img = Image.open("scratch/current_spotify.png")
# Flip vertically so it's right side up for human coordinate inspection
img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
img_flipped.save("scratch/spotify_right_way_up.png")
print("Saved right way up image:", img_flipped.size)
