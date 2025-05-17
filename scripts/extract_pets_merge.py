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

    target_folder = PetImagesFolder(target_path)
    petid_to_all_images = defaultdict(list)

    for source in args.src:
        source_folder = PetImagesFolder(source)

        # ✅ processed.txt のマージ
        processed_path = source / "processed.txt"
        with open(target_path_processed, "at") as processed:
            with open(processed_path, "rt") as src_processed:
                copyfileobj(src_processed, processed)

        # ✅ train.data の読み込み → 画像パスだけ集める
        data_file = source / "train.data"
        with open(data_file, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                pet_id = record["pet_id"]
                all_paths = [record["source_path"]] + record["paths"]
                petid_to_all_images[pet_id].extend(all_paths)

        # ✅ 対象画像のコピー
        for idx in range(len(source_folder)):
            images, pet_id, source = source_folder.get_record(idx)
            target_folder.add_record(images, pet_id, source)

    # ✅ ペア形式に変換： anchor → others 形式
    output_data_file = target_path / "train.data"
    with open(output_data_file, "w", encoding="utf-8") as f:
        for pet_id, paths in petid_to_all_images.items():
            if len(paths) < 2:
                continue
            anchor = paths[0]
            pairs = [[anchor, p] for p in paths[1:]]

            json.dump({"pet_id": pet_id, "paths": pairs}, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    target_folder.save_info()
    print(f"✅ Merged pair-form train.data written to: {output_data_file}")
