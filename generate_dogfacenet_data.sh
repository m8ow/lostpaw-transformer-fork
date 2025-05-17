#!/bin/bash

OUTPUT_FILE="./output/data/train.data"
BASE_DIR="./output/data/images"

rm -f "$OUTPUT_FILE"
mkdir -p tmp_pet_map

# グルーピング：画像を pet_id ごとに分類
for f in ${BASE_DIR}/*.jpg; do
    filename=$(basename "$f")
    pet_id=$(echo "$filename" | cut -d'.' -f1)
    echo "$f" >> "tmp_pet_map/$pet_id.txt"
done

# 各 pet_id に対して連続ペアを作成 [[img1, img2], [img2, img3], ...]
for pet_file in tmp_pet_map/*.txt; do
    pet_id=$(basename "$pet_file" .txt)
    mapfile -t images < "$pet_file"
    num_images=${#images[@]}

    if [ "$num_images" -ge 2 ]; then
        group_list=""
        for ((i = 0; i < num_images - 1; i++)); do
            img1="${images[$i]}"
            img2="${images[$((i + 1))]}"
            group_list+="[\"$img1\", \"$img2\"],"
        done
        # Remove trailing comma
        group_list=${group_list%,}
        echo "{\"pet_id\": \"$pet_id\", \"paths\": [${group_list}]}" >> "$OUTPUT_FILE"
    fi
done

rm -r tmp_pet_map
echo "✅ train.data (paired group format) written to $OUTPUT_FILE"