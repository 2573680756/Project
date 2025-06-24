import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# 假设你已经定义好了 VisionTransformer 并且可以导入
from vit_model import VisionTransformer  # 替换为你的模块路径
from ResNet import ResNet50
from EfficientNet import EfficientNet
from EfficientNet_V2 import EfficientNetV2

# 参数设置
DATA_DIR = "/app/zwy/FFSSD-ResNet-master/datasets/plantvillage dataset/color"  # 替换为实际路径
BATCH_SIZE = 32
VAL_RATIO = 0.2  # 验证集比例
SEED = 42  # 随机种子（确保可复现）

# 定义数据增强和归一化
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载完整数据集
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

# 计算训练集和验证集大小
val_size = int(VAL_RATIO * len(full_dataset))
train_size = len(full_dataset) - val_size

# 划分数据集（固定随机种子）
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# 替换验证集的transform（禁用数据增强）
val_dataset.dataset.transform = val_transform

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 计算类别数量
num_classes = len(full_dataset.classes)
'''
# 初始化模型
model = VisionTransformer(img_size=224,
                          patch_size=16,
                          embed_dim=1024,
                          depth=24,
                          num_heads=16,
                          representation_size=None,
                          num_classes=num_classes)

# model=ResNet()
'''
model=EfficientNetV2(num_classes=num_classes)

# 将模型移到 GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

# 定义训练和验证的轮数
NUM_EPOCHS = 10

# 初始化记录最佳验证损失和对应的模型
best_val_loss = float("inf")
best_model_path = "best_model.pth"

# 初始化 TensorBoard 记录器
writer = SummaryWriter()

# 初始化列表以保存训练和验证的损失和准确率
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 训练过程
for epoch in range(NUM_EPOCHS):
    model.train()  # 切换到训练模式
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # 计算平均训练损失和准确率
    avg_train_loss = running_train_loss / len(train_loader)
    train_accuracy = 100.0 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # 验证过程
    model.eval()  # 切换到评估模式
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # 计算平均验证损失和准确率
    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100.0 * correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    writer.add_scalar("Loss/validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # 保存最优模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved better model with validation loss: {avg_val_loss:.4f}")

# 关闭 TensorBoard 记录器
writer.close()

print("Training complete. Best model saved to", best_model_path)

# 绘制损失函数和准确率图像
plt.figure(figsize=(12, 5))

# 绘制损失函数
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')  # 保存图像到文件
plt.show()