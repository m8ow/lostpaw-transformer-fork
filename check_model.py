import torch
from pathlib import Path
from lostpaw.model.model import PetViTContrastiveModel

# ====== 設定 ======
MODEL_PATH = Path("output/models/model_2025_05_15_183913.pt")
MODEL_DIR = Path("output/models")  # ViT encoderやconfigが保存されてるフォルダ
LATENT_DIM = 128

# ====== モデル初期化 ======
print("🔧 モデルを初期化します...")
model = PetViTContrastiveModel(model_path=MODEL_DIR, output_dim=LATENT_DIM, device="cpu")

# ====== .ptファイルの中身チェック ======
print(f"\n📂 モデルファイル内容チェック: {MODEL_PATH}")
state = torch.load(MODEL_PATH, map_location="cpu")

if isinstance(state, dict):
    print("✅ このファイルは state_dict です")
    print(f"🔍 キー数: {len(state.keys())}")
    print(f"🔑 最初のキー: {list(state.keys())[0]}")
else:
    print("⚠️ これはモデルオブジェクト本体のようです（state_dict ではありません）")
    exit()

# ====== state_dict ロード確認 ======
print("\n📥 モデルに state_dict を読み込みます...")
result = model.load_state_dict(state, strict=False)
print("✅ load_state_dict 結果:")
print(result)

# ====== 重みの統計確認 ======
print("\n📊 重みの統計確認:")
for name, param in model.named_parameters():
    mean = param.data.mean().item()
    std = param.data.std().item()
    print(f"🔸 {name:50s} → 平均: {mean:.6f}, 標準偏差: {std:.6f}")
    break  # 最初の1つだけ表示（全表示したいなら break を削除）
