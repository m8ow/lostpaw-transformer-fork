# 使い方
```bash
# download DogFaceNet dataset
mkdir ./output
wget https://zenodo.org/records/12578449/files/DogFaceNet_alignment.zip -O ./output/raw-data.zip
unzip ./output/raw-data.zip -d ./output/ && rm ./output/raw-data.zip
# unzipない場合: `python3 -c "import zipfile, os; z='./output/raw-data.zip'; zipfile.ZipFile(z).extractall('./output/raw-data/'); os.remove(z)"`

# build
docker buildx build --load -t lostpaw-transformer .

# mount & run
docker run -it \
  --gpus all \
  --dns 8.8.8.8 \
  --name lostpaw-transformer-dev \
  -v $(pwd):/app -w /app \
  lostpaw-transformer bash

# init pip
pip install --upgrade pip
pip install -e .
pip install "wandb==0.15.12" "pydantic<2.0"

## PetFaceNetのデータセットを変換
python convert_petface.py

# pre-treatment
## 重複データを削除
python scripts/clean_dataset.py output/raw-data
## 壊れたデータを削除
python pick_broken_image.py output/raw-data

## raw-data.jsonlを作成
python generate_dogfacenet_data.py

## ペット画像の切り出しとAugmentation
huggingface-cli login # tokenを自分で発行して、入力してください
python scripts/extract_pets.py \
  --info_file ./output/raw-data/raw-data.jsonl \
  --model_path ./output/weights \
  --output_dir ./output/generated \
  --threads 4 \
  --batch_size 4

## マルチスレッド出力のマージ（必要に応じて）
python scripts/extract_pets_merge.py \
  output/generated/thread_* \
  output/data

# train
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -c lostpaw/configs/default.yaml

# run again
docker start -ai lostpaw-transformer-dev
docker exec -it lostpaw-transformer-dev bash
```

# LostPaw: Finding Lost Pets using a Contrastive Learning Transformer

This project is a study on the use of artificial intelligence in identifying lost pets. Specifically, we trained a contrastive neural network model to differentiate between pictures of dogs and evaluated its performance on a held-out test set. The goal of this project is to explore the potential of AI in the search for lost pets and to discuss the implications of these results. The project includes code for the model training and evaluation, but not the data used in the study. However, we do provide the scripts used to acquire the data.

# Requirements
The following dependencies are required to run this project:

* Python 3.7 or higher
* Torch 1.13 or higher 
* NumPy
* Pandas
* Transformers
* Timm
* Wandb
* Tqdm
* Pillow

# Getting Started

To get started with this project, you can clone the repository to your local machine and install the required dependencies using pip:

```bash
pip install --upgrade pip
pip install -e .
```

Once the dependencies are installed, you can run the code to train and evaluate the model (make sure the data is available):

```bash
python scripts/train.py -c lostpaw/configs/default.yaml
```

# Results
![accuracy](./docs/figures/accuracy.png)

* Achieved an average F1-score of 88.8% on the cross-validation set.
* Validation accuracy closely followed the train accuracy, indicating no overfitting.
* The loss value steadily dropped from 1.16 to 0.04 during training suggesting the model learned an effective representation of the data.
* These results suggest the contrastive learning model can make accurate decisions on unseen samples.

# Dataset
As we do not have the rights to publish the data, we recommend that the user obtains their own images and places them in a desired folder. In this case, please update the `info_path` flags during training. This can either be done by supplying additional flags to the training script, or by creating a new config.yaml file:
```
python scripts/train.py -c lostpaw/configs/default.yaml --info_path "output/data/train.data"
```

The `info_path` flag requires a file that contains rows of image entries, which have a format similar to the following:

```jsonl
{"pet_id": "35846622", "paths": ["output/data/35846622/0.jpg", "output/data/35846622/1.jpg", "output/data/35846622/2.jpg"]}
{"pet_id": "35846624", "paths": ["output/data/35846624/0.jpg", "output/data/35846624/1.jpg", "output/data/35846624/2.jpg"]}
``` 

The `paths` key can contain as many image paths as you desire, where each path should point to a different augmentation of the same image. For different images per pet include multiple entries with the same `pet_id`.


# Resources
A project created for the "High Tech Systems and Materials" Honours Master's track at the University of Groningen.

* Encoding of pet images: [ViT Base, patch 16, size 384](https://huggingface.co/google/vit-base-patch16-384)
* Pet recognition: [DETR ResNet 50](https://huggingface.co/facebook/detr-resnet-50)
* [Contrastive loss](https://ieeexplore.ieee.org/abstract/document/1640964)
