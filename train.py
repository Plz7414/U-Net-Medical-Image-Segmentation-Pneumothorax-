#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#pip install opencv-python pandas matplotlib tqdm
#pip3 install torchinfo==1.8.0
#pip install segmentation-models-pytorch
#作者:丁仲恩
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ==========================================
# 參數設定
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 20
IMG_SIZE = (256, 256)
DATA_DIR = "siim-acr-pneumothorax"


# ==========================================
# 視覺化函式
# ==========================================
def visualize_results(model, val_loader, device, epoch, num_samples=3):
    """
    從驗證集中抽取樣本，畫出原圖、真值 (GT) 與預測值 (Pred)
    """
    model.eval()
    imgs, masks = next(iter(val_loader))  # 取得一個 batch
    imgs, masks = imgs.to(device), masks.to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(imgs))
        preds = (preds > 0.5).float()  # 二值化

    # 轉回 CPU 進行繪圖
    imgs = imgs.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(min(num_samples, len(imgs))):
        # 1. 原始圖片
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.title(f"Original X-Ray (Epoch {epoch + 1})")
        plt.axis('off')

        # 2. 真實標記 (Ground Truth)
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.imshow(masks[i][0], cmap='Reds', alpha=0.4)  # 用紅色透明覆蓋
        plt.title("Ground Truth (Pneumothorax)")
        plt.axis('off')

        # 3. 模型預測 (Prediction)
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.imshow(preds[i][0], cmap='Blues', alpha=0.4)  # 用藍色透明覆蓋
        plt.title("Model Prediction")
        plt.axis('off')

    plt.tight_layout()
    save_path = f"result_epoch_{epoch + 1}.png"
    plt.savefig(save_path)
    print(f"--- 已儲存視覺化結果至 {save_path} ---")
    plt.close()  # 關閉視窗避免佔用記憶體


# ==========================================
# Dataset
# ==========================================
class PneumoDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, transform=None):
        self.data = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        name = img_name if img_name.endswith('.png') else f"{img_name}.png"
        img_path = os.path.join(self.img_dir, name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.uint8)
        image = cv2.resize(image, IMG_SIZE)

        mask_path = os.path.join(self.mask_dir, name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, IMG_SIZE) if mask is not None else np.zeros(IMG_SIZE, dtype=np.float32)
        else:
            mask = np.zeros(IMG_SIZE, dtype=np.float32)

        mask = (mask > 0).astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(mask).unsqueeze(0)


def calculate_metrics(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0: return 1.0
    return (intersection / (union + 1e-7)).item()


# ==========================================
# 主訓練流程
# ==========================================
def main():
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    csv_path = os.path.join(DATA_DIR, "stage_1_train_images.csv")
    all_df = pd.read_csv(csv_path)

    print("正在分析資料集以進行平衡抽樣...")
    pos_list, neg_list = [], []
    mask_dir = os.path.join(DATA_DIR, "png_masks")

    for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
        img_id = row[0]
        mask_path = os.path.join(mask_dir, img_id if img_id.endswith('.png') else f"{img_id}.png")
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and np.sum(m) > 0:
                pos_list.append(row)
                continue
        neg_list.append(row)

    pos_df, neg_df = pd.DataFrame(pos_list), pd.DataFrame(neg_list)
    print(f"發現患病樣本: {len(pos_df)}, 健康樣本: {len(neg_df)}")

    if len(pos_df) == 0:
        print("錯誤：找不到任何患病樣本！")
        return

    n_samples = min(len(pos_df), len(neg_df))
    balanced_df = pd.concat([pos_df.sample(n_samples, random_state=42),
                             neg_df.sample(n_samples, random_state=42)]).sample(frac=1).reset_index(drop=True)

    split = int(0.8 * len(balanced_df))
    train_loader = DataLoader(
        PneumoDataset(balanced_df.iloc[:split], os.path.join(DATA_DIR, "png_images"), mask_dir, transform),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        PneumoDataset(balanced_df.iloc[split:], os.path.join(DATA_DIR, "png_images"), mask_dir, transform),
        batch_size=BATCH_SIZE, shuffle=False)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_dice = smp.losses.DiceLoss(mode='binary')
    criterion_focal = smp.losses.FocalLoss(mode='binary')

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_iou = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = torch.sigmoid(model(imgs))
            loss = 0.7 * criterion_dice(outputs, masks) + 0.3 * criterion_focal(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += calculate_metrics(outputs, masks)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # 驗證
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs, v_masks = v_imgs.to(DEVICE), v_masks.to(DEVICE)
                v_outputs = torch.sigmoid(model(v_imgs))
                val_iou += calculate_metrics(v_outputs, v_masks)

        print(
            f"Epoch [{epoch + 1}] Train IoU: {train_iou / len(train_loader):.4f} | Val IoU: {val_iou / len(val_loader):.4f}")

        # 每隔幾代儲存一次視覺化圖片 (例如每代都存，或設定 epoch % 2 == 0)
        visualize_results(model, val_loader, DEVICE, epoch)

    torch.save(model.state_dict(), "final_pneumo_model.pth")
    print("訓練完成，模型與圖片已儲存。")


if __name__ == "__main__":
    main()