"""
PetFaceデータセット変換ユーティリティ

このスクリプトは以下の機能を提供します：
1. 入力フォルダ内のPNGファイルを再帰的に検索
2. ディレクトリ名を接頭辞とした新しいファイル名で出力フォルダにコピー
3. フォルダ構造を平坦化してファイル管理を簡素化

使用方法:
    python convert_petface.py input_folder output_folder
"""

import sys
from pathlib import Path
import shutil

# 引数チェック
if len(sys.argv) != 3:
    print("Usage: python script.py input_folder output_folder")
    sys.exit(1)

input_folder = Path(sys.argv[1])
output_folder = Path(sys.argv[2])
output_folder.mkdir(parents=True, exist_ok=True)

# PNGファイルを再帰的に検索してコピー
for filepath in input_folder.rglob("*.png"):
    rel_path = filepath.relative_to(input_folder)
    folder_name = rel_path.parent.name  # 最も内側のディレクトリ
    new_filename = f"{folder_name}.{filepath.name}"
    target_path = output_folder / new_filename
    shutil.copy2(filepath, target_path)
    print(f"✅ Copied: {filepath} → {target_path}")
