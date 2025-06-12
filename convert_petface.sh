#!/bin/bash

# エラーハンドリング
set -e

# 引数のチェック
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_folder output_folder"
  exit 1
fi

input_folder="$1"
output_folder="$2"

# 出力ディレクトリを作成（存在しない場合）
mkdir -p "$output_folder"

# 再帰的にPNGファイルを処理
find "$input_folder" -type f -name "*.png" | while read -r filepath; do
  # 入力フォルダからの相対パスを取得
  rel_path="${filepath#$input_folder/}"

  # ディレクトリとファイル名に分割
  folder_path=$(dirname "$rel_path")
  filename=$(basename "$rel_path")

  # フォルダ名の最後の部分を抽出（最も内側のディレクトリ名）
  folder_name=$(basename "$folder_path")

  # 出力ファイルパスを組み立て
  output_file="${output_folder}/${folder_name}.${filename}"

  # ファイルをコピー
  cp "$filepath" "$output_file"
done
