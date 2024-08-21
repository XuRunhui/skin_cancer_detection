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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import h5py
import io
from sklearn.model_selection import train_test_split

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




class CustomImageDataset(Dataset):
    def __init__(self, csv_file, hdf5_file, transform=None, mode='train', train_data=None):
        # self.img_dir = img_dir
        self.mode = mode
        if self.mode == "train":
            print("train data init begin")
        elif self.mode == "val":
            print("val data init begin")

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

        if mode == 'train':
            self.annotations = self.get_full_dataframe(csv_file)
            self.encode_labels()
            self.normalize_numerical_data()
        elif mode == 'val' or mode == 'test' and train_data is not None:
            print("getting full dataframe")
            self.annotations = self.get_full_dataframe(csv_file, train_data=train_data)
            print("encoding labels")
            self.encode_labels(train_data)
            print("normalizing numerical data")
            self.normalize_numerical_data(train_data)


        if self.mode == "train":
            print("train data init done")
        elif self.mode == "val":
            print("val data init done")
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
            isic_id = self.annotations.iloc[idx]["isic_id"]

            if isic_id in f:
                image = f[isic_id]
                # Check if the data is numerical before conversion
                image_data = image[()]
                # 将字节字符串解码为图像
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                

        if self.transform:
            image = self.transform(image)

        # Load categorical data

        numerical_df = self.annotations[self.numerical_vars]
        categorical_df = self.annotations[self.categorical_vars]
        numerical_df = numerical_df.iloc[idx]
        categorical_df = categorical_df.iloc[idx]

        numerical_data = torch.tensor(numerical_df.values, dtype=torch.float)
        categorical_data = torch.tensor(categorical_df.values, dtype=torch.long)

        # Load numerical data
        if self.mode in ['train', 'val']:
            label = torch.tensor(int(self.annotations.iloc[idx]["target"]))
            return image, categorical_data, numerical_data, label

        elif self.mode == 'test':
            isic_id = self.annotations.iloc[idx]["isic_id"]
            return image, categorical_data, numerical_data, isic_id
        

    def get_full_dataframe(self, df, train_data=None):
        def fill_missing_with_distribution(series, distribution):
            missing_indices = series[series.isna()].index
            filled_values = np.random.choice(distribution.index, size=len(missing_indices), p=distribution.values)
            series.loc[missing_indices] = filled_values
            return series
        
        if self.mode == 'train':
            df['lesion_id'] = df['lesion_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
        elif self.mode != 'train':
            for category in ['lesion_id', 'mel_mitotic_index']:
                dis = train_data.annotations[category].value_counts(normalize=True)
                generated_lesion_ids = np.random.choice(dis.index, size=len(df), p=dis.values)
                df[category] = generated_lesion_ids

        for category in ['sex', 'anatom_site_general']:
            dis = df[category].value_counts(normalize=True)
            df[category] = fill_missing_with_distribution(df[category], dis)
        
        mean_age = df['age_approx'].mean()
        df['age_approx'] = df['age_approx'].fillna(mean_age)
        
        if self.mode == 'train':
            df['iddx_2'] = df['iddx_2'].fillna(df['iddx_1'])
            df['iddx_3'] = df['iddx_3'].fillna(df['iddx_2'])
            df['iddx_4'] = df['iddx_4'].fillna(df['iddx_3'])
            df['iddx_5'] = df['iddx_5'].fillna(df['iddx_4'])
        elif self.mode != 'train':
            dis = train_data.annotations['iddx_full'].value_counts(normalize=True)
            generated_lesion_ids = np.random.choice(dis.index, size=len(df), p=dis.values)
            df['iddx_full'] = generated_lesion_ids
            for d in ['iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5']:
                df[d] = df['iddx_full']
        if self.mode != 'train':
            tbp_lv_dnn_lesion_confidence_mean = train_data.annotations['tbp_lv_dnn_lesion_confidence'].mean()
            tbp_lv_dnn_lesion_confidence_std = train_data.annotations['tbp_lv_dnn_lesion_confidence'].std()

            df['tbp_lv_dnn_lesion_confidence'] = np.random.normal(loc=tbp_lv_dnn_lesion_confidence_mean, scale=tbp_lv_dnn_lesion_confidence_std, size = len(df))

        return df
    
    def encode_labels(self, train_data=None):
        if self.mode == 'train':
            for col in self.categorical_vars:
                le = LabelEncoder()
                self.annotations[col] = le.fit_transform(self.annotations[col])
                self.label_encoders[col] = le
        elif self.mode != 'train' and train_data is not None:
            for col in self.categorical_vars:
                le = train_data.label_encoders[col]
                mask = ~self.annotations[col].isin(le.classes_)
    
                # 对新标签进行随机替换
                if mask.any():
                    self.annotations.loc[mask, col] = np.random.choice(
                        le.classes_, size=mask.sum(), p=train_data.annotations[col].value_counts(normalize=True).values
                    )
                
                # 进行编码
                self.annotations[col] = le.transform(self.annotations[col])
                print(f"Label encoder for {col} has {len(le.classes_)} classes")

                # self.annotations[col] = self.annotations[col].apply(
                #     lambda x: x if x in le.classes_ else np.random.choice(le.classes_, p=train_data.annotations[col].value_counts(normalize=True).values)
                # )
                # self.annotations[col] = le.transform(self.annotations[col])
                # print(f"Label encoder for {col} has {len(le.classes_)} classes")

    def normalize_numerical_data(self, train_data=None):
        if self.mode == 'train':
            scaler = MinMaxScaler()
            self.annotations[self.numerical_vars] = scaler.fit_transform(self.annotations[self.numerical_vars])
        elif self.mode != 'train' and train_data is not None:
            scaler = MinMaxScaler()
            scaler.fit(train_data.annotations[train_data.numerical_vars])
            self.annotations[self.numerical_vars] = scaler.transform(self.annotations[self.numerical_vars])




def generate_output(model_path):
    mytransform = get_transform()
    csv_file = pd.read_csv("../data/train-metadata.csv")
    train_df, val_df = train_test_split(csv_file, test_size=0.2, random_state=42)


    train_set = CustomImageDataset(csv_file=train_df, hdf5_file="../data/train-image.hdf5", transform=mytransform)
    val_set = CustomImageDataset(csv_file=val_df, hdf5_file="../data/train-image.hdf5", transform=mytransform, mode='val', train_data=train_set)
    test_set = CustomImageDataset(csv_file=pd.read_csv("../data/test-metadata.csv"), hdf5_file="../data/test-image.hdf5", transform=mytransform, mode='test', train_data=train_set)


    categorical_dims = [2, 5, 1, 2, 21, 8, 7, 3, 2, 52, 3, 15, 28, 52, 52, 7]
    model = CombinedModel(2, categorical_dims, 35)  # load your model here

    # 加载 state_dict
    state_dict = torch.load("best_model.pt")

    # 创建一个新的 state_dict，将键名中的 'module.' 前缀移除
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # 移除 'module.' 前缀
        new_state_dict[new_key] = v

    # 加载新的 state_dict 到模型
    model.load_state_dict(new_state_dict)


    model.eval()
    model = model.cuda()
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    results = []

    with torch.no_grad():
        results = []
        for images, categorical_data, numerical_data, isic_ids in tqdm(test_loader):
            images = images.cuda()
            categorical_data = categorical_data.cuda()
            numerical_data = numerical_data.cuda()
            
            outputs = model(images, categorical_data, numerical_data)
            probabilities = F.softmax(outputs, dim=1)
            
            for isic_id, p in zip(isic_ids, probabilities):
                # Assuming binary classification, we take the probability of the positive class (index 1)
                results.append({'isic_id': isic_id, 'target': p[1].item()})

    # Save results to a CSV file
    output_df = pd.DataFrame(results)
    output_df.to_csv("model_output_new.csv", index=False)
    print("Output saved to model_output.csv")


# Example usage
generate_output("best_model.pt")
    