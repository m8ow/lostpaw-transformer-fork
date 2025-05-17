from PIL import Image
import os

image_dir = "/app/output/data/images"
bad_files = []

for fname in os.listdir(image_dir):
    path = os.path.join(image_dir, fname)
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except Exception as e:
        print(f"Corrupt image: {fname} ({e})")
        bad_files.append(fname)

print(f"Found {len(bad_files)} broken images.")
