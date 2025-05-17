import torch
from PIL import Image
import argparse
import os
from pathlib import Path

# あなたのモデルクラス
from lostpaw.model.model import PetViTContrastiveModel


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    return img  # PILのまま渡す


def load_model(model_weights_path, model_dir, latent_dim=128):
    model = PetViTContrastiveModel(
        model_path=Path(model_dir),  # ViTのconfigとencoder保存場所
        output_dim=latent_dim,
        device="cpu"
    )
    model.load_model(model_weights_path)
    model.eval()
    return model


def compare_images(img1_path, img2_path, model, threshold=0.85):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    with torch.no_grad():
        z1 = model(img1)
        z2 = model(img2)

    print("z1:", z1.numpy())
    print("z2:", z2.numpy())
    print("difference:", (z1 - z2).abs().max().item())

    similarity = torch.nn.functional.cosine_similarity(z1, z2).item()
    print(f"\n🧠 Cosine Similarity: {similarity:.4f}")

    if similarity > threshold:
        print("✅ 同一犬個体の可能性が高い")
    else:
        print("❌ 異なる犬の可能性が高い")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two dog images")
    parser.add_argument("img1", type=str, help="Path to first image")
    parser.add_argument("img2", type=str, help="Path to second image")
    parser.add_argument("--model", type=str, default="output/models/model_2025_05_15_183913.pt", help="Path to model weights")
    parser.add_argument("--model_dir", type=str, default="output/models", help="Path to directory containing encoder and model subfolders")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent space size")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold for match")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"\n🔍 Comparing:\n- {args.img1}\n- {args.img2}")
    model = load_model(args.model, args.model_dir, latent_dim=args.latent_dim)
    compare_images(args.img1, args.img2, model, threshold=args.threshold)
