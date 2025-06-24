import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
train_dataset, val_dataset = torch.utils.data.random_split(
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

# 加载模型
from vit_model import VisionTransformer  # 替换为你的模块路径
from vgg import VGG16
from ResNet import ResNet50
from EfficientNet import EfficientNet
from EfficientNet_V2 import EfficientNetV2
num_classes = len(full_dataset.classes)
'''
model = VisionTransformer(img_size=224,
                          patch_size=16,
                          embed_dim=1024,
                          depth=24,
                          num_heads=16,
                          representation_size=None,
                          num_classes=num_classes)
'''
model=EfficientNetV2(num_classes=num_classes)

# 加载训练好的模型参数
best_model_path = "/app/zwy/best_model.pth"  # 替换为你的模型路径
model.load_state_dict(torch.load(best_model_path))
# 将模型移到 GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 测试过程
model.eval()  # 切换到评估模式
running_val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算平均验证损失和准确率
avg_val_loss = running_val_loss / len(val_loader)
accuracy = 100.0 * correct / total

print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.2f}%")