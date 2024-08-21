import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 销毁分布式环境
def cleanup():
    dist.destroy_process_group()

# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["isic_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.annotations.iloc[idx]["target"]))

        if self.transform:
            image = self.transform(image)

        return image, label

# 定义transform函数
def get_transform():
    transform = transforms.Compose([
        transforms.Resize(600, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(600),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def train(rank, world_size):
    setup(rank, world_size)

    # 数据预处理
    transform = get_transform()
    dataset = CustomImageDataset(csv_file="../data/train-metadata.csv", img_dir="../data/train-image/image", transform=transform)

    # 数据集拆分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建分布式数据加载器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, sampler=val_sampler)

    # 检查是否有可用的 GPU
    device = torch.device(f'cuda:{rank}')

    # 加载预训练的 EfficientNetB7 模型
    model = EfficientNet.from_pretrained('efficientnet-b0')

    # 冻结卷积基
    for param in model.parameters():
        param.requires_grad = False

    # 修改分类器以适应你的任务，添加额外的全连接层
    num_classes = 2  # 设置为你的分类数
    model._fc = nn.Sequential(
        nn.Linear(model._fc.in_features, 512),  # 新添加的全连接层
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)  # 最终分类层
    )
    
    for param in model._fc.parameters():
        param.requires_grad = True
    # 将模型移到 GPU 上
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model._fc.parameters(), lr=0.001)
    # 包装模型
    model = DDP(model, device_ids=[rank])

    # 训练模型
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 验证模型
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)