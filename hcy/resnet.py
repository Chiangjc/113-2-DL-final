import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import wandb  # 新增：引入 wandb

class YOLOMultilabelDataset(Dataset):
    """
    處理 YOLO 格式的資料，轉換為 multi-label classification
    YOLO 格式：class_id center_x center_y width height (normalized)
    資料夾結構:
    - images/train/, images/val/, images/test/
    - labels/train/, labels/val/, labels/test/
    """
    def __init__(self, images_dir, labels_dir, transform=None, class_names=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # 獲取所有圖片檔案
        self.image_files = [f for f in os.listdir(self.images_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()

        # 如果沒有提供 class_names，就從所有標籤檔裡提取
        if class_names is None:
            self.class_names = self._extract_class_names()
        else:
            self.class_names = class_names

        self.num_classes = len(self.class_names)
        # 處理所有標籤
        self.labels = self._process_labels()

        print(f"Dataset 初始化完成:")
        print(f"- 圖片目錄: {images_dir}")
        print(f"- 標籤目錄: {labels_dir}")
        print(f"- 圖片數量: {len(self.image_files)}")
        print(f"- 類別數量: {self.num_classes}")
        print(f"- 類別列表: {self.class_names}")

    def _extract_class_names(self):
        """從所有標籤檔案中提取 class_id，並建立 class name 列表"""
        all_class_ids = set()
        for img_file in self.image_files:
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            all_class_ids.add(class_id)
        # 如果沒有其他資訊，就以 class_{i} 命名
        class_names = [f'class_{i}' for i in sorted(all_class_ids)]
        return class_names

    def _process_labels(self):
        """把每張圖片的 YOLO 標籤轉成 multi-label 向量"""
        labels = []
        for img_file in self.image_files:
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            img_labels = torch.zeros(self.num_classes, dtype=torch.float32)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            class_id = int(parts[0])
                            if 0 <= class_id < self.num_classes:
                                img_labels[class_id] = 1.0
            labels.append(img_labels)
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = self.labels[idx]
        return image, labels

def create_data_transforms():
    """建立訓練/驗證的 transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class ResNetMultilabel(nn.Module):
    """基於 ResNet 的 multi-label 分類模型"""
    def __init__(self, num_classes, pretrained=True, model_name='resnet50', dropout_rate=0.5):
        super(ResNetMultilabel, self).__init__()

        # 載入預訓練的 ResNet 變體
        if model_name == 'resnet18':
            backbone = torchvision.models.resnet18(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet34':
            backbone = torchvision.models.resnet34(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            num_features = 2048
        elif model_name == 'resnet101':
            backbone = torchvision.models.resnet101(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"不支援的模型: {model_name}")

        # 移除原本分類層，只保留到 avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # 新增自訂的 multi-label 分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        print(f"模型初始化完成: {model_name}, 預訓練: {pretrained}")
        print(f"特徵維度: {num_features}, 類別數: {num_classes}")

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        output = self.classifier(features)
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=25, device='cuda', patience=10):
    """訓練模型並記錄到 wandb"""
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # --- Training phase ---
        model.train()
        running_loss = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / num_batches
        train_losses.append(train_loss)

        # --- Validation phase ---
        model.eval()
        val_running_loss = 0.0
        val_num_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                val_num_batches += 1

        val_loss = val_running_loss / val_num_batches
        val_losses.append(val_loss)

        # 更新學習率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss:   {val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # 將本 epoch 的損失與 LR 紀錄到 wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr
        })

        # Early stopping 檢查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_multilabel_model.pt')
            print('保存最佳模型')
            # 並上傳最新的最佳模型作為 wandb artifact（可選）
            wandb.save('best_multilabel_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    return train_losses, val_losses

def evaluate_model(model, test_loader, class_names, device='cuda', threshold=0.5):
    """模型評估並記錄指標到 wandb"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # sigmoid 轉機率
            probs = torch.sigmoid(outputs)
            predictions = (probs > threshold).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    print("\n=== 模型評估結果 ===")

    class_metrics = []
    for i, class_name in enumerate(class_names):
        y_true = all_labels[:, i]
        y_pred = all_predictions[:, i]

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == 1)

        class_metrics.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })

        print(f"{class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")

    # 計算 macro F1
    macro_f1 = np.mean([m['f1'] for m in class_metrics])

    # 計算 micro 指標
    total_tp = np.sum([np.sum((all_labels[:, i] == 1) & (all_predictions[:, i] == 1))
                       for i in range(len(class_names))])
    total_fp = np.sum([np.sum((all_labels[:, i] == 0) & (all_predictions[:, i] == 1))
                       for i in range(len(class_names))])
    total_fn = np.sum([np.sum((all_labels[:, i] == 1) & (all_predictions[:, i] == 0))
                       for i in range(len(class_names))])

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print(f"\n=== 整體性能 ===")
    print(f"Macro F1:      {macro_f1:.3f}")
    print(f"Micro Precision: {micro_precision:.3f}")
    print(f"Micro Recall:    {micro_recall:.3f}")
    print(f"Micro F1:        {micro_f1:.3f}")

    # 將各種評估指標上傳到 wandb
    wandb.log({
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    })

    # 也可以將每個 class 的 precision/recall/f1 逐一上傳
    for m in class_metrics:
        wandb.log({
            f"{m['class']}_precision": m['precision'],
            f"{m['class']}_recall": m['recall'],
            f"{m['class']}_f1": m['f1'],
            f"{m['class']}_support": m['support']
        })

    return all_predictions, all_labels, all_probs, class_metrics

def plot_training_history(train_losses, val_losses, output_dir):
    """繪製訓練歷史並存檔，然後上傳到 wandb"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    png_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 上傳圖到 wandb
    wandb.log({"training_history": wandb.Image(png_path)})
    print(f"訓練歷史圖已存至: {png_path}")

def analyze_class_distribution(images_dirs, labels_dirs, class_names, split_names):
    """分析各資料集的類別分布（此處不額外上傳到 wandb，可視需求自行加入）"""
    print("=== 類別分布分析 ===")
    for images_dir, labels_dir, split_name in zip(images_dirs, labels_dirs, split_names):
        print(f"\n{split_name} 集合:")
        class_counts = defaultdict(int)
        total_images = 0
        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in image_files:
            total_images += 1
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            if 0 <= class_id < len(class_names):
                                class_counts[class_id] += 1
        print(f"總圖片數: {total_images}")
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / total_images) * 100
            print(f"  {class_names[class_id]}: {count} ({percentage:.1f}%)")

# ----------------------------------------------------------------------------------------------------
# 主要執行程式
def main():
    import argparse

    parser = argparse.ArgumentParser(description='多標籤圖像分類訓練 (已整合 wandb)')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='模型架構')
    parser.add_argument('--images-dir', type=str, default='images', help='圖片目錄路徑')
    parser.add_argument('--labels-dir', type=str, default='labels', help='標籤目錄路徑')
    parser.add_argument('--output-dir', type=str, default='outputs', help='輸出目錄路徑')
    parser.add_argument('--resume', type=str, default='', help='恢復訓練的模型路徑')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout 率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='權重衰減')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping 耐心值')
    parser.add_argument('--threshold', type=float, default=0.5, help='分類閾值')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用預訓練權重')
    parser.add_argument('--no-pretrained', action='store_true', help='不使用預訓練權重')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['auto', 'cuda', 'cpu'], help='運算設備')
    parser.add_argument('--project-name', type=str, default='multilabel_resnet',
                        help='wandb 專案名稱')
    parser.add_argument('--run-name', type=str, default=None,
                        help='wandb run 名稱 (如果不提供，wandb 會自動產生)')
    args = parser.parse_args()

    # 處理 pretrained 參數
    if args.no_pretrained:
        args.pretrained = False

    # 設定運算設備
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)

    print(f"使用設備: {DEVICE}")
    print(f"模型架構: {args.model}")
    print(f"使用預訓練權重: {args.pretrained}")

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 wandb
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config={  # 將所有參數自動記錄在 wandb.config
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'model': args.model,
            'dropout_rate': args.dropout,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'threshold': args.threshold,
            'pretrained': args.pretrained,
            'device': str(DEVICE)
        }
    )
    wandb.config.update(args)  # 確保所有 args 都被記錄

    # 由 args.images-dir 和 args.labels-dir 設定 train/val/test 的子目錄
    TRAIN_IMAGES_DIR = os.path.join(args.images_dir, 'train')
    TRAIN_LABELS_DIR = os.path.join(args.labels_dir, 'train')
    VAL_IMAGES_DIR = os.path.join(args.images_dir, 'val')
    VAL_LABELS_DIR = os.path.join(args.labels_dir, 'val')
    TEST_IMAGES_DIR = os.path.join(args.images_dir, 'test')
    TEST_LABELS_DIR = os.path.join(args.labels_dir, 'test')

    # 創建 transforms
    train_transform, val_transform = create_data_transforms()

    # 載入資料集
    print("正在載入訓練資料...")
    train_dataset = YOLOMultilabelDataset(
        images_dir=TRAIN_IMAGES_DIR,
        labels_dir=TRAIN_LABELS_DIR,
        transform=train_transform
    )
    print("正在載入驗證資料...")
    val_dataset = YOLOMultilabelDataset(
        images_dir=VAL_IMAGES_DIR,
        labels_dir=VAL_LABELS_DIR,
        transform=val_transform,
        class_names=train_dataset.class_names  # 同樣的 class_names
    )
    print("正在載入測試資料...")
    test_dataset = YOLOMultilabelDataset(
        images_dir=TEST_IMAGES_DIR,
        labels_dir=TEST_LABELS_DIR,
        transform=val_transform,
        class_names=train_dataset.class_names
    )

    # 分析類別分布（可視需求選擇要不要上傳到 wandb）
    analyze_class_distribution(
        [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR],
        [TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR],
        train_dataset.class_names,
        ['Train', 'Validation', 'Test']
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # 建立模型
    model = ResNetMultilabel(
        num_classes=train_dataset.num_classes,
        pretrained=args.pretrained,
        model_name=args.model,
        dropout_rate=args.dropout
    ).to(DEVICE)

    # 讓 wandb 監看模型：自動記錄權重與梯度
    wandb.watch(model, log='all', log_freq=10)

    # 如果要從 checkpoint 恢復
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"從 {args.resume} 恢復模型...")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"從第 {start_epoch} 輪開始訓練")
        else:
            model.load_state_dict(checkpoint)
            print("僅載入模型權重完成")

    # 定義 loss、optimizer、scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    print("\n開始訓練...")
    print(f"模型: {args.model}")
    print(f"批次大小: {args.batch_size}")
    print(f"學習率: {args.learning_rate}")
    print(f"最大訓練輪數: {args.epochs}")
    print(f"Early stopping 耐心值: {args.patience}")

    # 執行訓練
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.epochs, device=DEVICE, patience=args.patience
    )

    # 把最終的 best model 移到輸出目錄
    model_save_path = os.path.join(args.output_dir, 'best_multilabel_model.pt')
    if os.path.exists('best_multilabel_model.pt'):
        os.replace('best_multilabel_model.pt', model_save_path)

    # 保存訓練歷史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'model': args.model,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': len(train_losses),
            'pretrained': args.pretrained,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay
        }
    }
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"訓練歷史已保存到: {history_path}")

    # 繪圖並上傳到 wandb
    plot_training_history(train_losses, val_losses, args.output_dir)

    # 載入最佳模型並評估
    print("載入最佳模型進行評估...")
    model.load_state_dict(torch.load(model_save_path))
    predictions, labels, probs, class_metrics = evaluate_model(
        model, test_loader, train_dataset.class_names, DEVICE, threshold=args.threshold
    )

    # 保存評估結果
    eval_results = {
        'class_metrics': class_metrics,
        'threshold': args.threshold,
        'class_names': train_dataset.class_names
    }
    eval_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"評估結果已保存到: {eval_path}")

    print(f"\n訓練與評估全部完成！")
    print(f"結果保存在: {args.output_dir}")
    print(f"最佳模型: {model_save_path}")
    print(f"訓練歷史: {history_path}")
    print(f"評估結果: {eval_path}")

    # 選擇性：結束時上傳最佳模型為 artifact
    artifact = wandb.Artifact('best_multilabel_model', type='model')
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)
    print("已上傳最佳模型到 W&B artifact")

    # 關閉 wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

# 額外工具函數（同原始碼一樣無需改動）

# def predict_single_image(model, image_path, class_names, transform, device='cuda', threshold=0.5):
#     """對單張圖片進行預測"""
#     model.eval()

#     image = Image.open(image_path).convert('RGB')
#     input_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(input_tensor)
#         probs = torch.sigmoid(output).cpu().numpy()[0]
#         predictions = (probs > threshold).astype(int)

#     print(f"圖片: {image_path}")
#     predicted_classes = []
#     for i, (class_name, prob, pred) in enumerate(zip(class_names, probs, predictions)):
#         if pred == 1:
#             predicted_classes.append(class_name)
#             print(f"  {class_name}: {prob:.3f}")

#     if not predicted_classes:
#         print("  沒有檢測到任何類別")

#     return predicted_classes, probs
