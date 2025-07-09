"""モデル評価・テストスクリプト

学習済みペット顔認識モデルをテストデータで評価し、精度、適合率、再現率、
F1スコアなどの様々な指標を計算します。テストバッチを処理し、性能統計をレポートします。

主な機能:
1. バッチベースのモデル評価
2. 複数の評価指標の計算
3. 設定可能なテストバッチサイズと数
4. 学習済みモデルチェックポイントとの互換性

使用方法:
    python test.py -c config.yaml
    python test.py -c config.yaml --test_batch_size 16
"""

from pathlib import Path
from pprint import pprint
from lostpaw.config.args import get_args
from lostpaw.data.data_folder import PetImagesFolder
from lostpaw.model.trainer import Trainer, TrainConfig
from tqdm import tqdm

import numpy as np

def main(args):
    config = TrainConfig(**vars(args))
    config.use_wandb = False

    trainer = Trainer(config)
    pet_data = trainer.pet_data
    batch_size = config.test_batch_size
    test_batches = pet_data.get_batches(batch_size=batch_size)

    metrics = np.zeros(4)
    for _ in tqdm(range(config.test_batch_count)):
        imgs1, imgs2, labels = next(test_batches)
        metrics += trainer.test_batch(imgs1, imgs2, labels, batch_size)

    m_diff, m_err1, m_err2, m_same = metrics / config.test_batch_count

    pprint(dict(m_diff=m_diff, m_err1=m_err1, m_err2=m_err2, m_same=m_same))


if __name__ == "__main__":
    args = get_args()

    main(args)
