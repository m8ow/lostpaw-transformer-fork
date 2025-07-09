"""ペット顔認識モデルのメイン学習スクリプト

クロスバリデーションを使用してペット顔認識のための対比学習トランスフォーマーモデルを学習します。
YAML設定ファイルとコマンドライン引数で設定可能なハイパーパラメータをサポートします。

主な機能:
1. 設定可能なk-foldでのクロスバリデーション学習
2. YAMLベースの設定システム
3. 実験追跡のためのWeights & Biases統合
4. モデルチェックポイントと再開
5. 設定可能な損失マージンでの対比学習

使用方法:
    python train.py -c config.yaml
    python train.py -c config.yaml --cross_validation_k_fold 5
"""

from lostpaw.config.args import get_args
from lostpaw.model.trainer import Trainer, TrainConfig

def main(args):
    config = TrainConfig(**vars(args))

    k_folds = max(config.cross_validiton_k_fold, 1)
    pet_data = None

    for _ in range(k_folds):
        trainer = Trainer(config, pet_data)
        pet_data = trainer.pet_data

        trainer.train()
        
        pet_data.next_fold()

if __name__ == "__main__":
    args = get_args()

    main(args)