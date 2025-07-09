"""
マルチスレッドペット抽出出力統合スクリプト

複数のextract_pets.pyスレッドの出力を単一の統合データセットに結合します。
処理済み画像、メタデータを統合し、モデル学習に必要な形式の最終学習データファイルを作成します。

主な機能:
1. 複数のスレッド出力を単一のデータセットに統合
2. 全スレッドのprocessed.txtファイルを統合
3. 画像ファイルとメタデータを結合
4. 対比学習用のペア形式で最終train.dataファイルを作成

使用方法:
    python extract_pets_merge.py output/generated/thread_* output/data
"""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfileobj
import json
from collections import defaultdict

from lostpaw.data.data_folder import PetImagesFolder

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src", nargs="+", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    target_path = args.target
    target_path.mkdir(parents=True, exist_ok=True)
    target_path_processed = target_path / "processed.txt"
    target_path_processed.touch(exist_ok=True)

    # ✅ ペア形式 train.data とは別ファイルに info を保存するようにする
    target_folder = PetImagesFolder(target_path, info_file_name="images.info.json")
    petid_to_all_images = defaultdict(list)

    for source in args.src:
        source_folder = PetImagesFolder(source)

        # ✅ processed.txt のマージ
        processed_path = source / "processed.txt"
        with open(target_path_processed, "at") as processed:
            with open(processed_path, "rt") as src_processed:
                copyfileobj(src_processed, processed)

        # ✅ train.data の読み込み
        data_file = source / "train.data"
        with open(data_file, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pet_id = record["pet_id"]

                if "source_path" in record:
                    all_paths = [record["source_path"]] + record["paths"]
                else:
                    all_paths = record["paths"]

                petid_to_all_images[pet_id].extend(all_paths)

        # ✅ 画像をコピー
        for idx in range(len(source_folder)):
            images, pet_id, source = source_folder.get_record(idx)
            target_folder.add_record(images, pet_id, source)

    # ✅ 情報を別ファイルに保存（上書き回避）
    target_folder.save_info()

    # ✅ ペア形式の train.data を最後に出力（これが本命！）
    output_data_file = target_path / "train.data"
    with open(output_data_file, "w", encoding="utf-8") as f:
        for pet_id, paths in petid_to_all_images.items():
            if len(paths) < 2:
                continue
            anchor = paths[0]
            pairs = [[anchor, p] for p in paths[1:]]
            json.dump({"pet_id": pet_id, "paths": pairs}, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    print(f"✅ Merged pair-form train.data written to: {output_data_file}")
