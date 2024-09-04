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
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from torchvision.models import efficientnet_b0, efficientnet_b7

# class Discriminator(nn.Module):
#     def __init__(self, ngpu, nc = 3, ndf = 64):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)

#         return output.view(-1, 1).squeeze(1)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         torch.nn.init.normal_(m.weight, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         torch.nn.init.normal_(m.weight, 1.0, 0.02)
#         torch.nn.init.zeros_(m.bias)



class CombinedModel(nn.Module):
    def __init__(self, num_classes, categorical_dims, num_numerical_features):
        super(CombinedModel, self).__init__()
        # self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')


        model = efficientnet_b0(pretrained=False) 
        model.load_state_dict(torch.load('./efficientnet_b0.pth'))


        # model = efficientnet_b7(pretrained=False) 
        # model.load_state_dict(torch.load('./efficientnet_b7.pth'))
        self.efficientnet = model
        # 冻结 EfficientNet 的卷积层


        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # 分类器部分
        # in_features = self.efficientnet._fc.in_features 
        # self.efficientnet._fc = nn.Identity()  # 移除原来的全连接层

        in_features = self.efficientnet.classifier[1].in_features
        # in_features = 1
        # 移除原来的全连接层
        self.efficientnet.classifier = nn.Identity()


        # netD = Discriminator(1)
        # netD.apply(weights_init)

        # self.model = netD
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
    
# def get_transform():
#     transform = transforms.Compose([
#         transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return transform

def get_transform():
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.RandomRotation(30),  # 随机旋转 -30 到 30 度
        transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 以 50% 概率垂直翻转
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),  # 随机改变亮度、对比度和饱和度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    def __init__(self, csv_file, hdf5_file, fake_data = None, img_dir = None, transform=None, mode='train', train_data=None):
        # self.img_dir = img_dir
        self.fake_data = fake_data
        self.img_dir = img_dir
        self.mode = mode
        if self.mode == "train":
            print("train data init begin")
        elif self.mode == "val":
            print("val data init begin")

        self.transform = transform
        self.hdf5_file = h5py.File(hdf5_file, 'r')
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
        if self.fake_data is None:
            return len(self.annotations)
        else:
            return len(self.annotations) + len(self.fake_data)

    def __getitem__(self, idx):
        # Load image
        # img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["isic_id"] + ".jpg")
        # image = Image.open(img_path).convert("RGB")

        if idx < len(self.annotations):
            isic_id = self.annotations.iloc[idx]["isic_id"]

            if isic_id in self.hdf5_file:
                image = self.hdf5_file[isic_id]
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
            
        else:        
            fake_idx = idx - len(self.annotations)

            img_path = os.path.join(self.img_dir, f"{fake_idx + 1}.jpg")
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            numerical_df = self.fake_data[self.numerical_vars]
            categorical_df = self.fake_data[self.categorical_vars]
            numerical_df = numerical_df.iloc[fake_idx]
            categorical_df = categorical_df.iloc[fake_idx]

            numerical_data = torch.tensor(numerical_df.values, dtype=torch.float)
            categorical_data = torch.tensor(categorical_df.values, dtype=torch.long)

            # Load numerical data
            if self.mode in ['train', 'val']:
                label = torch.tensor(int(self.fake_data.iloc[fake_idx]["target"]))
                return image, categorical_data, numerical_data, label


        

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

# class CustomImageDataset(Dataset):
#     def __init__(self, csv_file, hdf5_file, transform=None, mode='train', train_data=None):
#         # self.img_dir = img_dir
#         self.mode = mode
#         if self.mode == "train":
#             print("train data init begin")
#         elif self.mode == "val":
#             print("val data init begin")

#         self.transform = transform
#         self.hdf5_file = hdf5_file
#         self.label_encoders = {}
        
#         self.categorical_vars = ['sex', 'anatom_site_general',
#                              'image_type', 'tbp_tile_type', 'tbp_lv_location', 
#                              'tbp_lv_location_simple', 'attribution', 'copyright_license', 
#                              'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 
#                              'iddx_4', 'iddx_5', 'mel_mitotic_index']
        
#         self.numerical_vars = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 
#                           'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 
#                           'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
#                           'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 
#                           'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 
#                           'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 
#                           'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 
#                           'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 
#                           'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 
#                           'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 
#                           'tbp_lv_y', 'tbp_lv_z', 'tbp_lv_dnn_lesion_confidence']

#         if mode == 'train':
#             self.annotations = self.get_full_dataframe(csv_file)
#             self.encode_labels()
#             self.normalize_numerical_data()
#         elif mode == 'val' and train_data is not None:
#             print("getting full dataframe")
#             self.annotations = self.get_full_dataframe(csv_file, train_data=train_data)
#             print("encoding labels")
#             self.encode_labels(train_data)
#             print("normalizing numerical data")
#             self.normalize_numerical_data(train_data)


#         if self.mode == "train":
#             print("train data init done")
#         elif self.mode == "val":
#             print("val data init done")
#         # print(numerical_df.isnull().sum())
#         # print(categorical_df.isnull().sum())
#         # print(numerical_df.values.dtype)
#         # print(categorical_df.values.dtype)

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         # Load image
#         # img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]["isic_id"] + ".jpg")
#         # image = Image.open(img_path).convert("RGB")

#         with h5py.File(self.hdf5_file, 'r') as f:
#             isic_id = self.annotations.iloc[idx]["isic_id"]

#             if isic_id in f:
#                 image = f[isic_id]
#                 # Check if the data is numerical before conversion
#                 image_data = image[()]
#                 # 将字节字符串解码为图像
#                 image = Image.open(io.BytesIO(image_data)).convert("RGB")
                

#         if self.transform:
#             image = self.transform(image)

#         # Load label
#         label = torch.tensor(int(self.annotations.iloc[idx]["target"]))

#         # Load categorical data

#         numerical_df = self.annotations[self.numerical_vars]
#         categorical_df = self.annotations[self.categorical_vars]
#         numerical_df = numerical_df.iloc[idx]
#         categorical_df = categorical_df.iloc[idx]

#         numerical_data = torch.tensor(numerical_df.values, dtype=torch.float)
#         categorical_data = torch.tensor(categorical_df.values, dtype=torch.long)

#         # Load numerical data
        

#         return image, categorical_data, numerical_data, label


#     def get_full_dataframe(self, df, train_data=None):
#         def fill_missing_with_distribution(series, distribution):
#             missing_indices = series[series.isna()].index
#             filled_values = np.random.choice(distribution.index, size=len(missing_indices), p=distribution.values)
#             series.loc[missing_indices] = filled_values
#             return series
        
#         if self.mode == 'train':
#             df['lesion_id'] = df['lesion_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
#         elif self.mode == 'val':
#             for category in ['lesion_id', 'mel_mitotic_index']:
#                 dis = train_data.annotations[category].value_counts(normalize=True)
#                 generated_lesion_ids = np.random.choice(dis.index, size=len(df), p=dis.values)
#                 df[category] = generated_lesion_ids

#         for category in ['sex', 'anatom_site_general']:
#             dis = df[category].value_counts(normalize=True)
#             df[category] = fill_missing_with_distribution(df[category], dis)
        
#         mean_age = df['age_approx'].mean()
#         df['age_approx'] = df['age_approx'].fillna(mean_age)
        
#         if self.mode == 'train':
#             df['iddx_2'] = df['iddx_2'].fillna(df['iddx_1'])
#             df['iddx_3'] = df['iddx_3'].fillna(df['iddx_2'])
#             df['iddx_4'] = df['iddx_4'].fillna(df['iddx_3'])
#             df['iddx_5'] = df['iddx_5'].fillna(df['iddx_4'])
#         elif self.mode == 'val':
#             dis = train_data.annotations['iddx_full'].value_counts(normalize=True)
#             generated_lesion_ids = np.random.choice(dis.index, size=len(df), p=dis.values)
#             df['iddx_full'] = generated_lesion_ids
#             for d in ['iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5']:
#                 df[d] = df['iddx_full']
#         if self.mode == 'val':
#             tbp_lv_dnn_lesion_confidence_mean = train_data.annotations['tbp_lv_dnn_lesion_confidence'].mean()
#             tbp_lv_dnn_lesion_confidence_std = train_data.annotations['tbp_lv_dnn_lesion_confidence'].std()

#             df['tbp_lv_dnn_lesion_confidence'] = np.random.normal(loc=tbp_lv_dnn_lesion_confidence_mean, scale=tbp_lv_dnn_lesion_confidence_std, size = len(df))

#         return df
    
#     def encode_labels(self, train_data=None):
#         if self.mode == 'train':
#             for col in self.categorical_vars:
#                 le = LabelEncoder()
#                 self.annotations[col] = le.fit_transform(self.annotations[col])
#                 self.label_encoders[col] = le
#         elif self.mode == 'val' and train_data is not None:
#             for col in self.categorical_vars:
#                 le = train_data.label_encoders[col]
#                 mask = ~self.annotations[col].isin(le.classes_)
    
#                 # 对新标签进行随机替换
#                 if mask.any():
#                     self.annotations.loc[mask, col] = np.random.choice(
#                         le.classes_, size=mask.sum(), p=train_data.annotations[col].value_counts(normalize=True).values
#                     )
                
#                 # 进行编码
#                 self.annotations[col] = le.transform(self.annotations[col])
#                 print(f"Label encoder for {col} has {len(le.classes_)} classes")

#                 # self.annotations[col] = self.annotations[col].apply(
#                 #     lambda x: x if x in le.classes_ else np.random.choice(le.classes_, p=train_data.annotations[col].value_counts(normalize=True).values)
#                 # )
#                 # self.annotations[col] = le.transform(self.annotations[col])
#                 # print(f"Label encoder for {col} has {len(le.classes_)} classes")

#     def normalize_numerical_data(self, train_data=None):
#         if self.mode == 'train':
#             scaler = MinMaxScaler()
#             self.annotations[self.numerical_vars] = scaler.fit_transform(self.annotations[self.numerical_vars])
#         elif self.mode == 'val' and train_data is not None:
#             scaler = MinMaxScaler()
#             scaler.fit(train_data.annotations[train_data.numerical_vars])
#             self.annotations[self.numerical_vars] = scaler.transform(self.annotations[self.numerical_vars])




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
        best_loss = float('inf')
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.total_epochs = total_epochs
        self.best_loss = best_loss

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

    # def _validate(self, epoch):
    #     self.model.eval()
    #     val_loss = 0.0
    #     correct = 0
    #     with torch.no_grad():
    #         for images, categorical_data, numerical_data, labels in tqdm(self.val_data, desc=f"Validating Epoch {epoch+1}/{self.total_epochs}"):
    #             images, categorical_data, numerical_data, labels = (
    #                 images.to(self.gpu_id),
    #                 categorical_data.to(self.gpu_id),
    #                 numerical_data.to(self.gpu_id),
    #                 labels.to(self.gpu_id)
    #             )
    #             outputs = self.model(images, categorical_data, numerical_data)
    #             loss = self.criterion(outputs, labels)
    #             val_loss += loss.item() * images.size(0)
    #             _, preds = torch.max(outputs, 1)
    #             correct += (preds == labels).sum().item()

    #     val_loss /= len(self.val_data.dataset)
    #     accuracy = correct / len(self.val_data.dataset)
    #     print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    #     if val_loss < self.best_loss:
    #         self.best_loss = val_loss
    #         best_model_wts = self.model.state_dict()
    #         torch.save(best_model_wts, 'best_model.pt')
    #         print(f"New best model found at epoch {epoch+1} with validation loss: {val_loss:.4f}")

    def _validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        all_labels = []
        all_probs = []
        pos = 0
        true_pos = 0

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

                # Collect predictions and true labels
                probabilities = F.softmax(outputs, dim=1)
                all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probabilities of the positive class
                all_labels.extend(labels.cpu().numpy())
                
                _, preds = torch.max(outputs, 1)
                
                # Update true positive and positive counts
                true_pos += np.sum((labels.cpu().numpy() == 1) & (preds.cpu().numpy() == 1))
                pos += np.sum(labels.cpu().numpy() == 1)
                
                # Update the number of correct predictions
                correct += (preds == labels).sum().item()

# At this point, `val_loss`, `true_pos`, `pos`, and `correct` will hold the accumulated values for the entire validation set.


        val_loss /= len(self.val_data.dataset)
        accuracy = correct / len(self.val_data.dataset)

        # Calculate pAUC
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        pAUC_normalized = compute_pauc_above_tpr(all_labels, all_probs, tpr_threshold=0.8)

        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, pAUC: {pAUC_normalized:.4f}')
        print(f'True positives: {true_pos}, Positives: {pos}')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            best_model_wts = self.model.state_dict()
            torch.save(best_model_wts, 'best_model.pt')
            print(f"New best model found at epoch {epoch+1} with validation loss: {val_loss:.4f}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = f"checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch + 1} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)



def compute_pauc_above_tpr(y_true, y_scores, tpr_threshold=0.8):
    # Step 1: Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Step 2: Filter out TPR < 0.8
    indices_above_tpr = np.where(tpr >= tpr_threshold)[0]
    
    # If no TPR values are above the threshold, return 0
    if len(indices_above_tpr) == 0:
        return 0.0
    
    # Select the portion of the curve where TPR >= 0.8
    fpr_above = fpr[indices_above_tpr]
    tpr_above = tpr[indices_above_tpr]
    
    # Step 3: Calculate pAUC using the trapezoidal rule
    pauc = np.trapz(tpr_above, fpr_above)
    
    # Normalize the pAUC by dividing by the maximum possible pAUC in this range
    max_pauc = 0.2  # Because TPR range is from 0.8 to 1, and max FPR range would be 0 to 1
    pauc_normalized = pauc / max_pauc
    return pauc
    

def load_train_objs():
    mytransform = get_transform()
    csv_file = pd.read_csv("../data/train-metadata.csv")
    train_df, val_df = train_test_split(csv_file, test_size=0.2, random_state=42)


    train_df_majority = train_df[train_df.target==0]
    train_df_minority = train_df[train_df.target==1]
    train_df_minority_upsampled = resample(train_df_minority, 
                                         replace=True,     # sample with replacement
                                         n_samples=len(train_df_majority),    # to match majority class
                                         random_state=42) # reproducible results
    train_df_upsampled = pd.concat([train_df_majority, train_df_minority_upsampled])
    train_df_upsampled = train_df_upsampled.sample(frac=1, random_state = 42).reset_index(drop=True)


    fake_data = pd.read_csv("../data/resampled_data.csv")
    train_set = CustomImageDataset(csv_file=train_df, hdf5_file="../data/train-image.hdf5", transform=mytransform, fake_data=fake_data, img_dir="../data/fake_images", mode='train')
    val_set = CustomImageDataset(csv_file=val_df, hdf5_file="../data/train-image.hdf5", transform=mytransform, mode='val', train_data=train_set)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    categorical_dims = [2, 5, 1, 2, 21, 8, 7, 3, 2, 52, 3, 15, 28, 52, 52, 7]
    model = CombinedModel(2, categorical_dims, 35)  # load your model
    model_path = 'best_model.pt'
    if os.path.exists(model_path):

        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path)

        # 创建一个新的 state_dict，将键名中的 'module.' 前缀移除
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # 移除 'module.' 前缀
            new_state_dict[new_key] = v

        # 加载新的 state_dict 到模型
        model.load_state_dict(new_state_dict, strict = False)
    else:
        print("No existing model found, starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return train_set, val_set, model, optimizer



def prepare_dataloader(train_dataset: Dataset, val_dataset: Dataset, batch_size: int, seed: int = 42):
    torch.manual_seed(seed)
    
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader



def main(device, total_epochs, save_every, batch_size):
    dist.init_process_group(backend='nccl', init_method='env://')
    
    train_set, val_set, model, optimizer = load_train_objs()
    
    # Move model to the correct device
    model = model.to(device)
    
    # Wrap the model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    
    train_data, val_data = prepare_dataloader(train_set, val_set, batch_size)
    class_weights = torch.tensor([0.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainer = Trainer(model, train_data, val_data, optimizer, criterion, device, total_epochs, save_every)
    trainer.train(total_epochs)
    
    dist.destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    # 获取local_rank
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == -1:
        raise ValueError("LOCAL_RANK not found in environment variables")
    
    device = local_rank
    torch.cuda.set_device(device)
    
    main(device, args.total_epochs, args.save_every, args.batch_size)
