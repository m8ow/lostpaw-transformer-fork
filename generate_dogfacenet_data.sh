#!/bin/bash

OUTPUT_FILE="./output/raw-data/converted.jsonl"
BASE_DIR="./output/raw-data"

rm -f "$OUTPUT_FILE"
mkdir -p tmp_pet_map

# グルーピング：画像を pet_id ごとに分類
for f in ${BASE_DIR}/*.jpg; do
    filename=$(basename "$f")
    pet_id=$(echo "$filename" | cut -d'.' -f1)
    echo "$f" >> "tmp_pet_map/$pet_id.txt"
done

# 各 pet_id に対して画像を1枚ずつ出力（PetImageDataset 用）
for pet_file in tmp_pet_map/*.txt; do
    pet_id=$(basename "$pet_file" .txt)
    while IFS= read -r img; do
        echo "{\"petId\": \"$pet_id\", \"savedPath\": \"$img\"}" >> "$OUTPUT_FILE"
    done < "$pet_file"
done

rm -r tmp_pet_map
echo "✅ converted.jsonl (1 image per line) written to $OUTPUT_FILE"
