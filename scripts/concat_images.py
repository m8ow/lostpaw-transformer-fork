"""
画像結合ユーティリティ

このスクリプトは以下の機能を提供します：
1. 複数の画像を水平方向に結合
2. グロブパターンによる画像ファイルの一括読み込み
3. 結合画像のPNG形式での保存

使用方法:
    python concat_images.py image1.jpg image2.jpg "folder/*.png"
"""

from glob import glob
import sys
from PIL import Image, ImageDraw


images = [Image.open(f) for a in sys.argv[1:] for f in glob(a)]

print("nr of images:", len(images))

padding = 5

result = Image.new('RGB', (sum(i.width + padding for i in images) - padding, max(i.height for i in images)))
result_draw = ImageDraw.Draw(result)
result_draw.rectangle((0, 0, result.width, result.height), "white")
x_offset = 0
for im in images:
  result.paste(im, (x_offset,0))
  x_offset += im.width + padding

result.save('result.png')
