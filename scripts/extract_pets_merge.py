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

    # 統合先 PetImagesFolder の作成
    target_folder = PetImagesFolder(target_path)

    # --- 統合されたレコードを保持するための辞書 ---
    petid_to_pairs = defaultdict(list)

    for source in args.src:
        source_folder = PetImagesFolder(source)

        # processed.txt のマージ
        processed_path = source / "processed.txt"
        with open(target_path_processed, "at") as processed:
            with open(processed_path, "rt") as src_processed:
                copyfileobj(src_processed, processed)

        # train.data の読み込みとペア構築
        data_file = source / "train.data"
        with open(data_file, "rt") as f:
            for line in f:
                record = json.loads(line)
                pet_id = record["pet_id"]
                source_path = record["source_path"]
                augmented_paths = record["paths"]

                for aug_path in augmented_paths:
                    petid_to_pairs[pet_id].append([source_path, aug_path])

        # 対象画像もマージ（必要に応じて）
        for idx in range(len(source_folder)):
            images, pet_id, source = source_folder.get_record(idx)
            target_folder.add_record(images, pet_id, source)

    # 結果を書き出し
    output_data_file = target_path / "train.data"
    with open(output_data_file, "w") as f:
        for pet_id, pairs in petid_to_pairs.items():
            json.dump({"pet_id": pet_id, "paths": pairs}, f, ensure_ascii=False)
            f.write("\n")

    target_folder.save_info()

    print(f"✅ Merged contrastive-style train.data written to: {output_data_file}")
