"""破損画像検出・削除ユーティリティ

データセット処理中の問題を防ぐために、ディレクトリ内の破損したまたは読み取り不可能な画像ファイルをスキャンして削除します。
整合性を検証するため、各画像をロードしてRGB形式に変換を試行します。

主な機能:
1. 画像ファイルのディレクトリスキャン
2. ロードとRGB変換による各画像のテスト
3. ロードや変換に失敗したファイルの削除
4. 詳細なエラーレポートの提供

使用方法:
    python pick_broken_image.py <image_directory>
"""

from PIL import Image
import os
from argparse import ArgumentParser

def remove_corrupt_images(image_dir):
    bad_files = []

    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)

        # ファイルでなければスキップ（例：サブディレクトリなど）
        if not os.path.isfile(path):
            continue

        try:
            with Image.open(path) as img:
                img.load()
                img.convert("RGB")
        except Exception as e:
            print(f"Removing corrupt image: {fname} ({e})")
            bad_files.append(fname)
            try:
                os.remove(path)
            except Exception as delete_error:
                print(f"Failed to delete {fname}: {delete_error}")

    print(f"Removed {len(bad_files)} broken images.")
    return bad_files


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Path to image folder")
    args = parser.parse_args()

    remove_corrupt_images(args.input_folder)
