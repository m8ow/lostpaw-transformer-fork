"""データセット可視化・検査ユーティリティ

検査と分析のためのデータセットの視覚的表現を生成します。
サンプルバッチ、データ分布、対比学習で使用されるペアリング例を示す画像グリッドを作成します。

主な機能:
1. 画像ペアでのバッチ可視化
2. データセット統計と分布分析
3. 品質保証のためのサンプル生成
4. 設定可能な出力形式とバッチサイズ

使用方法:
    python visualize_data.py <data_folder>
"""

import argparse
import os
from pathlib import Path

from lostpaw.data.data_folder import PetImagesFolder
from lostpaw.data.dataset import RandomPairDataset

if __name__ == "__main__":

    # Create a parser object
    parser = argparse.ArgumentParser()

    # Add an argument for the folder path
    parser.add_argument('folder', help='The path to the data folder')

    # Parse the arguments
    args = parser.parse_args()

    data_folder = PetImagesFolder(Path(args.folder), "train.data")

    # data_folder.describe(True)

    # exit(0)

    # Dataset
    pet_data = RandomPairDataset(data_folder, 0.5)

    for i, batch in enumerate(pet_data.get_batches(batch_size=3)):
        pet_data.visualize_batch(batch, file_name=f"data_{i}.jpg")
        if i == 2:
            break;

