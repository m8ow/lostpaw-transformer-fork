"""
DogFaceNet学習データ生成ユーティリティ

このスクリプトは以下の機能を提供します：
1. 指定ディレクトリ内の画像ファイル（jpg/png）を自動検出
2. ファイル名から犬のIDを抽出してグルーピング
3. JSONL形式でのデータセット情報ファイル生成

使用方法:
    python generate_dogfacenet_data.py
"""

import json
from pathlib import Path
from collections import defaultdict

base_dir = Path("../output/raw-data")
output_file = base_dir / "raw-data.jsonl"
tmp_map = defaultdict(list)

# グルーピングしながら即座にprint
print("📂 グルーピング中...")
for img_path in base_dir.glob("*.[jp][pn]g"):  # .jpg or .png
    pet_id = img_path.stem
    abs_path = img_path.resolve()
    tmp_map[pet_id].append(abs_path)
    print(f"  ➕ {pet_id} に追加: {abs_path}")

# JSONL出力
with open(output_file, "w") as f:
    print("\n📄 JSONL出力:")
    for pet_id, paths in tmp_map.items():
        for path in paths:
            record = {"petId": pet_id, "savedPath": str(path)}
            line = json.dumps(record)
            print(line)
            f.write(line + "\n")

print(f"\n✅ JSONLファイル作成完了: {output_file.resolve()}")
