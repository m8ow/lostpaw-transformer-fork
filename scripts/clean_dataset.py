"""
データセットクリーニングと学習/テスト分割ユーティリティ

このスクリプトは以下の機能を提供します：
1. ソースパスに基づいてデータセットから重複画像を削除
2. 設定可能な比率で学習用とテスト用データセットに分割
3. 分割後のペットID別画像分布の解析

使用方法:
    python clean_dataset.py <dataset_path> --deduplicate
    python clean_dataset.py <dataset_path> --split-to test.jsonl --split-percentage 0.1
"""

from argparse import ArgumentParser
from os import remove
from pathlib import Path
from random import random
import numpy as np
from lostpaw.data.data_folder import PetImagesFolder


def deduplicate(folder: PetImagesFolder):
    df = folder.data_frame()
    dups = df.duplicated(subset=["source"])
    deduplicated = df[~dups]
    duplicated = df[dups]

    for row in duplicated["paths"]:
        for path in row:
            remove(path)

    deduplicated.to_json(folder.info_file, orient="records", lines=True)


def split_test(folder: PetImagesFolder, split_name: str, test_percentage: float):
    df = folder.data_frame()
    ids = df["pet_id"].unique()
    test_pet_count = int(len(ids) * test_percentage)

    test_ids = set(np.random.choice(ids, size=test_pet_count, replace=False))

    is_test = df["pet_id"].map(lambda id: id in test_ids)
    test = df[is_test]
    train = df[~is_test]

    print("train:", np.bincount(train.groupby("pet_id")["paths"].apply(len), minlength=6).tolist())
    print("test:", np.bincount(test.groupby("pet_id")["paths"].apply(len), minlength=6).tolist())

    if input("should we continue? (y/n): ") == "y":
        train.to_json(folder.info_file, orient="records", lines=True)

        test_path = Path(folder.info_file)
        test_path.name = split_name
        test.to_json(test_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("path", type=Path, help="path to dataset")
    parser.add_argument(
        "--deduplicate", action="store_true", help="deduplicate the dataset"
    )
    parser.add_argument(
        "--split-to",
        type=str,
        help="when given split a fraction of the info file into a new info \
            file with the given name",
    )
    parser.add_argument(
        "--split-percentage",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    if args.deduplicate:
        deduplicate(PetImagesFolder(args.path))

    if args.split_to:
        split_test(PetImagesFolder(args.path), args.split_to, args.split_percentage)
