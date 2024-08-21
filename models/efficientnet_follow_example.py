import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import h5py

class CombinedModel(nn.Module):
    def __init__(self, num_classes, categorical_dims, num_numerical_features):
        super(CombinedModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 冻结 EfficientNet 的卷积层
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # 分类器部分
        in_features = self.efficientnet._fc.in_features 
        self.efficientnet._fc = nn.Identity()  # 移除原来的全连接层
        self.image_fc = nn.Linear(in_features, 512)
        
        # # 处理类别数据
        # self.categorical_embeddings = nn.ModuleList([
        #     nn.Embedding(num_embeddings=10, embedding_dim=5) for _ in range(num_categorical_features)
        # ])

        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim+1, embedding_dim=min(50, (dim + 1) // 2))
            for dim in categorical_dims
        ])

        categorical_total_dim = sum([embedding.embedding_dim for embedding in self.categorical_embeddings])

        self.categorical_fc = nn.Linear(categorical_total_dim, 32)
        
        # 处理数值数据
        self.numerical_fc = nn.Linear(num_numerical_features, 32)
        
        # 最终分类器
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 32 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, categorical_data, numerical_data):
        # 图像特征
        x_image = self.efficientnet(images)
        x_image = self.image_fc(x_image)
        
        # 类别特征
        # print(f"Shape of categorical_data: {categorical_data.shape}")
        x_categorical = [embedding(categorical_data[:, i]) for i, embedding in enumerate(self.categorical_embeddings)]
        # for i, tensor in enumerate(x_categorical):
        #     print(f"Shape of tensor {i}: {tensor.shape}")   

        x_categorical = torch.cat(x_categorical, dim=1)
        x_categorical = self.categorical_fc(x_categorical)
        
        # 数值特征
        x_numerical = self.numerical_fc(numerical_data)
        
        # 结合所有特征
        x = torch.cat((x_image, x_categorical, x_numerical), dim=1)
        x = self.final_fc(x)
        
        return x
    
def get_transform():
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_efficientnet():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_classes = 2

    model._fc = nn.Sequential(
        nn.Linear(model._fc.in_features, 512),  # 新添加的全连接层
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)  # 最终分类层
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in model._fc.parameters():
        param.requires_grad = True  
    return model



class CustomImageDataset(Dataset):
    def __init__(self, csv_file, hdf5_file, transform=None):
        # self.img_dir = img_dir
        self.transform = transform
        self.hdf5_file = hdf5_file
        self.label_encoders = {}
        self.categorical_vars = ['sex', 'anatom_site_general',
                             'image_type', 'tbp_tile_type', 'tbp_lv_location', 
                             'tbp_lv_location_simple', 'attribution', 'copyright_license', 
                             'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 
                             'iddx_4', 'iddx_5', 'mel_mitotic_index']
        
        self.numerical_vars = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 
                          'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 
                          'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
                          'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 
                          'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 
                          'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 
                          'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 
                          'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 
                          'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 
                          'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 
                          'tbp_lv_y', 'tbp_lv_z', 'tbp_lv_dnn_lesion_confidence']
        self.annotations = self.get_full_dataframe(csv_file)
        self.encode_labels()
        self.normalize_numerical_data()  

        # print(numerical_df.isnull().sum())
        # print(categorical_df.isnull().sum())
        # print(numerical_df.values.dtype)
        # print(categorical_df.values.dtype)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        # img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["isic_id"] + ".jpg")
        # image = Image.open(img_path).convert("RGB")

        with h5py.File(self.hdf5_file, 'r') as f:
            image = np.array(f[self.annotations.iloc[idx]["isic_id"]])
            image = Image.fromarray(image).convert("RGB")


        if self.transform:
            image = self.transform(image)

        # Load label
        label = torch.tensor(int(self.annotations.iloc[idx]["target"]))

        # Load categorical data

        numerical_df = self.annotations[self.numerical_vars]
        categorical_df = self.annotations[self.categorical_vars]
        numerical_df = numerical_df.iloc[idx]
        categorical_df = categorical_df.iloc[idx]

        numerical_data = torch.tensor(numerical_df.values, dtype=torch.float)
        categorical_data = torch.tensor(categorical_df.values, dtype=torch.long)

        # Load numerical data
        

        return image, categorical_data, numerical_data, label


    def get_full_dataframe(self, path):
        df = pd.read_csv(path)
        df['lesion_id'] = df['lesion_id'].apply(lambda x: 'Not Null' if pd.notnull(x) else 'Null')

        def fill_missing_with_distribution(series, distribution):
            missing_indices = series[series.isna()].index
            filled_values = np.random.choice(distribution.index, size=len(missing_indices), p=distribution.values)
            series.loc[missing_indices] = filled_values
            return series

        for category in ['sex', 'anatom_site_general']:
            dis = df[category].value_counts(normalize=True)
            df[category] = fill_missing_with_distribution(df[category], dis)
        
        mean_age = df['age_approx'].mean()
        df['age_approx'] = df['age_approx'].fillna(mean_age)
        df['lesion_id'] = df['lesion_id'].apply(lambda x: 1 if pd.notnull(x) else 0)

        df['iddx_2'] = df['iddx_2'].fillna(df['iddx_1'])
        df['iddx_3'] = df['iddx_3'].fillna(df['iddx_2'])
        df['iddx_4'] = df['iddx_4'].fillna(df['iddx_3'])
        df['iddx_5'] = df['iddx_5'].fillna(df['iddx_4'])

        return df
    
    def encode_labels(self):
        for col in self.categorical_vars:
            le = LabelEncoder()
            self.annotations[col] = le.fit_transform(self.annotations[col])
            self.label_encoders[col] = le


    def normalize_numerical_data(self):

        scaler = MinMaxScaler()

        self.annotations[self.numerical_vars] = scaler.fit_transform(self.annotations[self.numerical_vars])   

    def get_negative_sample(self):
        print(self.annotations[self.annotations['target'] == 0].value_counts())


# class CustomImageDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["isic_id"] + ".jpg")
#         image = Image.open(img_path).convert("RGB")
#         label = torch.tensor(int(self.annotations.iloc[idx]["target"]))

#         if self.transform:
#             image = self.transform(image)

#         return image, label

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        gpu_id: int,
        total_epochs: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.total_epochs = total_epochs
        self.best_loss = float('inf')

    def _run_batch(self, images, categorical_data, numerical_data, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images, categorical_data, numerical_data)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch+1}/{self.total_epochs}")

        self.model.train()
        running_loss = 0.0
        for images, categorical_data, numerical_data, labels in tqdm(self.train_data, desc=f"Epoch {epoch+1}/{self.total_epochs}"):
            images, categorical_data, numerical_data, labels = (
                images.to(self.gpu_id),
                categorical_data.to(self.gpu_id),
                numerical_data.to(self.gpu_id),
                labels.to(self.gpu_id)
            )
            loss = self._run_batch(images, categorical_data, numerical_data, labels)
            running_loss += loss * images.size(0)

        epoch_loss = running_loss / len(self.train_data.dataset)
        print(f'Epoch {epoch + 1}/{self.total_epochs}, Loss: {epoch_loss:.4f}')

        self._validate(epoch)

    def _validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, categorical_data, numerical_data, labels in tqdm(self.val_data, desc=f"Validating Epoch {epoch+1}/{self.total_epochs}"):
                images, categorical_data, numerical_data, labels = (
                    images.to(self.gpu_id),
                    categorical_data.to(self.gpu_id),
                    numerical_data.to(self.gpu_id),
                    labels.to(self.gpu_id)
                )
                outputs = self.model(images, categorical_data, numerical_data)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss /= len(self.val_data.dataset)
        accuracy = correct / len(self.val_data.dataset)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            best_model_wts = self.model.state_dict()
            torch.save(best_model_wts, 'best_model.pt')
            print(f"New best model found at epoch {epoch+1} with validation loss: {val_loss:.4f}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch + 1} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)



def load_train_objs():
    mytransform = get_transform()

    train_set = CustomImageDataset(csv_file="../data/train-metadata.csv", hdf5_file="../data/train-image.hdf5", transform=mytransform)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    categorical_dims = [2, 5, 1, 2, 21, 8, 7, 3, 2, 52, 3, 15, 28, 52, 52, 7]
    model = CombinedModel(2, categorical_dims, 35)  # load your model
    model_path = 'best_model.pt'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model found, starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return train_set, model, optimizer



def prepare_dataloader(dataset: Dataset, batch_size: int, seed: int = 42):
    # 设置随机种子
    torch.manual_seed(seed)

    # 数据集拆分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    target_zeros_count = 0

    # 遍历 val_dataset 统计 target 为 0 的数量
    for i in range(len(val_dataset)):
        print(i)
        _, _, _, target = val_dataset[i]  # 根据你的 __getitem__ 返回的内容调整此行代码
        if target == 1:
            target_zeros_count += 1
            print(f"Found target 0 at index {i}")

    print(f"Number of targets equal to 0 in validation dataset: {target_zeros_count}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader



def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data, val_data = prepare_dataloader(dataset, batch_size)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, train_data, val_data, optimizer,criterion, device, total_epochs, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)
