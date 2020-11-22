from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import time

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensor



#DATA_FILE_PATH = "Dataset/Dataset/label_data.csv"


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self,file_paths,  transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        start = time.time()
        file_path = self.file_paths[idx]

        bg    = Image.open(file_path[0])
        fg_bg = Image.open(file_path[1])
        mask  = Image.open(file_path[2])
        depth = Image.open(file_path[3])

        try:

            if self.transform:
                augmented_bg = self.transform(image=bg)
                image_bg = augmented_bg['image']
                augmented_fg_bg = self.transform(image=fg_bg)
                image_fg_bg = augmented_fg_bg['image']
                augmented_mask = self.transform(image=mask)
                image_mask = augmented_mask['image']
                augmented_depth = self.transform(image=depth)
                image_depth = augmented_depth['image']

        except:
            print('Error while transform:')

        images = {'bg': image_bg, 'fg_bg': image_fg_bg, 'mask': image_mask, 'depth': image_depth}

        end  = time.time()
        time_spent = (end-start)/60
        #print(f"{time_spent:.3} minutes")

        return images


def getfdatadf(data_file_path):
    #file_path = os.path.join(datafolder ,file_path)
    return pd.read_csv(data_file_path)

def loadalbumentationdata(datapath , ratio , batch_size , train_transform_list , test_transform_list):

    data_df = getfdatadf(datapath)

    test_Data_len = int(ratio * len(data_df))

    X_train =data_df[:test_Data_len]

    X_test =data_df[-test_Data_len:]

    def composetransormlist(transform_list):
        transform_list.append(ToTensor())
        return A.Compose(transform_list)


    albumentations_transform_train = composetransormlist(train_transform_list)

    albumentations_transform_test = composetransormlist(test_transform_list)

    albumentations_train_dataset = AlbumentationsDataset(
        file_paths=X_train,
        transform=albumentations_transform_train,
    )

    albumentations_test_dataset = AlbumentationsDataset(
        file_paths=X_test,
        transform=albumentations_transform_test,
    )

    train_loader = torch.utils.data.DataLoader(albumentations_train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(albumentations_test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)

    return train_loader, test_loader