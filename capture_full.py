import time
from capture_spotify import capture_window
from PIL import Image

capture_window(657680, "spot_full.bmp")
Image.open("spot_full.bmp").save("spot_full.png")
print("Saved spot_full.png")
