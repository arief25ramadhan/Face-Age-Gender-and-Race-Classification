# 1. Import library
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import glob
import os

## Just in case your images don't load properly
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# # 2. Download data
# !wget https://hydranets-data.s3.eu-west-3.amazonaws.com/UTKFace.zip
# !jar xf UTKFace.zip

# 2.1. Set image paths
image_paths = sorted(glob.glob("../UTKFace/*.jpg.chip.jpg"))
# print("Try one image path: ", image_paths[0])

# 3. Dataset class
class UTKFace(Dataset):
    def __init__(self, image_paths):
        # Mean and Std for ImageNet
        mean=[0.485, 0.456, 0.406] # ImageNet
        std=[0.229, 0.224, 0.225] # ImageNet

        # Define the Transforms
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Set Inputs and Labels
        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []

        for path in image_paths:
            filename = os.path.basename(path).split("_")
            if len(filename)==4:
                self.images.append(path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))
    
    def __len__(self):
         return len(self.images)

    def __getitem__(self, index):
        # Load an Image
        img = Image.open(self.images[index]).convert('RGB')
        # Transform it
        img = self.transform(img)

        # Get the Labels
        age = self.ages[index]
        gender = self.genders[index]
        race = self.races[index]
        
        # Return the sample of the dataset
        sample = {'image':img, 'age': age, 'gender': gender, 'race':race}
        return sample


# 4. Load data function
def load_data(image_paths=image_paths):

    TRAIN_SPLIT = 0.6
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2

    num_train = round(TRAIN_SPLIT*len(image_paths))
    num_val = round(VAL_SPLIT*len(image_paths))
    num_test = len(image_paths) - num_train - num_val

    print("Length of all data: ", len(image_paths))
    print("Length of train data: ", num_train)
    print("Length of val data: ", num_val)
    print("Length of test data: ", num_test)

    (train_dataset, val_dataset, test_dataset) = random_split(image_paths,[num_train, num_val, num_test],generator=torch.Generator().manual_seed(42))

    # Data Loader
    BATCH_SIZE = 64

    train_dataloader = DataLoader(UTKFace(train_dataset), shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(UTKFace(val_dataset), shuffle=False, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(UTKFace(test_dataset), shuffle=False, batch_size=BATCH_SIZE)

    return train_dataloader, val_dataloader, test_dataloader

train_dataloader, val_dataloader, test_dataloader = load_data(image_paths=image_paths)

# 5. Try
dataset_dict = {
    'race_id': {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'
    },
    'gender_id': { 0: 'Male', 1: 'Female'
    }
}

# sample = next(iter(train_dataloader))
# print(sample["image"][0].shape)
# print(sample["age"][0].item())
# print(dataset_dict['gender_id'][sample["gender"][0].item()])
# print(dataset_dict['race_id'][sample["race"][0].item()])