#!/bin/bash

OUTPUT_FILE="./output/raw-data.jsonl"
BASE_DIR="./output/raw-data"

rm -f "$OUTPUT_FILE"
mkdir -p tmp_pet_map

# グルーピング：画像を pet_id ごとに分類
for f in ${BASE_DIR}/*.jpg; do
    filename=$(basename "$f")
    pet_id=$(echo "$filename" | cut -d'.' -f1)
    echo "$f" >> "tmp_pet_map/$pet_id.txt"
done

# 各 pet_id に対して全画像を列挙
for pet_file in tmp_pet_map/*.txt; do
    pet_id=$(basename "$pet_file" .txt)
    mapfile -t images < "$pet_file"
    if [ "${#images[@]}" -gt 0 ]; then
        path_list=""
        for img in "${images[@]}"; do
            path_list+="\"$img\","
        done
        # Remove trailing comma
        path_list=${path_list%,}
        echo "{\"pet_id\": \"$pet_id\", \"paths\": [${path_list}]}" >> "$OUTPUT_FILE"
    fi
done

rm -r tmp_pet_map
echo "✅ train.data (all image paths per pet_id) written to $OUTPUT_FILE"