import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import datetime
from sklearn.preprocessing import StandardScaler

class VisibilityDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, scaler=None):
        """
        Args:
            image_dir (string): Path to the image folder.
            label_file (string): Path to the CSV file with labels and time-series data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_data = pd.read_csv(label_file)
        self.label_data['Time'] = pd.to_datetime(self.label_data['Time'], format='%Y/%m/%d %H:%M')
        self.label_data.ffill(inplace=True)  # Forward fill to handle missing values
        self.label_data['Month'] = self.label_data['Time'].dt.month
        self.label_data['Hour'] = self.label_data['Time'].dt.hour
        if scaler:
            self.label_data[['AOD', 'RH']] = scaler.transform(self.label_data[['AOD', 'RH']])
        self.transform = transform
        self.images = self._load_images()

    def _load_images(self):
        images = {}
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                exposure_type = filename[-5]
                timestamp = datetime.datetime.strptime(filename[:16], '%Y_%m_%d_%H_%M')
                key = (timestamp, exposure_type)
                image_path = os.path.join(self.image_dir, filename)
                images[key] = image_path
        return images

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        row = self.label_data.iloc[idx]
        time = row['Time']
        visibility = row['Visibility']
        aod = row['AOD']
        rh = row['RH']
        month = row['Month']
        Hour = row['Hour']

        # Find the closest image by timestamp and exposure
        img0_path = self.images.get((time, '0'))
        img1_path = self.images.get((time, '1'))

        image0 = Image.open(img0_path)
        image1 = Image.open(img1_path)

        if self.transform:
            image0 = self.transform(image0)
            image1 = self.transform(image1)

        sample = {'image0': image0, 'image1': image1, 'visibility': visibility, 
                  'aod': aod, 'rh': rh, 'month': month, 'Hour': Hour}

        return sample

def get_transform(target_size=(299, 299)):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

def get_scaler(train_file):
    data = pd.read_csv(train_file)
    # data.fillna(method='ffill', inplace=True)
    scaler = StandardScaler()
    scaler.fit(data[['AOD', 'RH']])
    return scaler

def get_dataloader(image_dir, label_file, batch_size=16,target_size=(299, 299)):
    transform = get_transform(target_size)
    scaler = get_scaler(label_file)
    dataset = VisibilityDataset(image_dir, label_file, transform=transform,scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Example usage
if __name__ == '__main__':
    train_loader = get_dataloader('./new_May_25/train_mask_rename', './new_May_25/train_labels.csv')
    test_loader = get_dataloader('./new_May_25/test_mask_rename', './new_May_25/test_labels.csv')
    for i, data in enumerate(train_loader):
        print(i, data['image0'].shape, data['visibility'].shape)
        # print(data)
        if i == 0:
            break