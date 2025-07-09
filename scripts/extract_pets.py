"""
ペット顔抽出とデータ拡張処理プロセッサ

このスクリプトは、DETRモデルを使用してフル画像からペットの顔を抽出し、
データ拡張を適用して学習用データセットを作成します。大規模データセットの
効率的なバッチ処理のためのマルチスレッド処理をサポートします。

主な機能:
1. DETR（Detection Transformer）を使用してペット顔を検出・抽出
2. データ拡張（回転、反転、色調整）の適用
3. スケーラビリティのためのマルチスレッド処理
4. 整理されたディレクトリ構造での抽出顔の保存

使用方法:
    python extract_pets.py --info_file data.jsonl --model_path ./models --output_dir ./output --threads 4
"""

from argparse import ArgumentParser, Namespace
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from glob import glob
import json
import logging
from pathlib import Path
from typing import Any, List, Set, Tuple
from multiprocessing import Process
import torch
from PIL.Image import Image

from lostpaw.data import PetImageDataset, DetrPetExtractor
from lostpaw.data.auto_augment import DataAugmenter
from lostpaw.data.extract_pets import lookup_next_image_name


def extract_images(
    data: PetImageDataset,
    output_dir: Path,
    model_path: Path,
    batch_size: int = 4,
    device: str = "cpu"
):
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    pet_extractor = DetrPetExtractor(model_path, device=device)
    pet_augment = DataAugmenter()

    with open(output_dir / "processed.txt", "at") as processed_file:
        with open(output_dir / "train.data", "at") as resulting_file:
            for batch in data.get_batches(batch_size=batch_size):
                input_images = batch["images"]
                input_labels = [str(l) for l in batch["labels"]]
                input_paths = batch["paths"]
                labels = zip(input_labels, input_paths)

                cropped: List[Tuple[Image, Tuple[str, Any]]] = pet_extractor.extract(
                    input_images, labels, output_size=(224, 224)
                )
                for image, (label, path) in cropped:
                    augmented = pet_augment.get_transforms(image, 2)
                    augmented.insert(0, image)  # 元画像を先頭に

                    paths = [save_image(image, label, output_dir) for image in augmented]

                    processed_file.write(f"{paths[0]}\n")  # 元画像

                    # ✅ 正しい形式で1行出力（JSONL）
                    resulting_file.write(json.dumps({
                        "pet_id": label,
                        "source_path": paths[0],  # 元画像
                        "paths": paths[1:]        # augmented画像のみ
                    }, ensure_ascii=False))
                    resulting_file.write("\n")

                processed_file.flush()
                resulting_file.flush()


def save_image(image, label, output_dir):
    folder_path = Path(output_dir, str(label))
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    path = lookup_next_image_name(folder_path)
    image.save(path)
    return str(path.resolve())


def main(args: Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_file_paths = glob(str(output_dir / "**" / "processed.txt"), recursive=True)
    ignore: Set[str] = set()
    for processed_file_path in processed_file_paths:
        with open(processed_file_path, "rt") as processed_file:
            ignore.update(l.strip() for l in processed_file.readlines())

    pet_data = PetImageDataset.load_from_file(Path(args.info_file), ignore=ignore)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    processes: List[Process] = []
    for i, data_subset in enumerate(pet_data.split(args.threads)):
        sub_out_path = output_dir / f"thread_{i}"
        sub_out_path.mkdir(exist_ok=True)
        process = Process(
            target=extract_images,
            args=[data_subset, sub_out_path, args.model_path, args.batch_size, device]
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--info_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    main(args)
