import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PolypDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir (str): Path to the data directory containing 'images' and 'masks' subdirectories.
            csv_file (str): Path to the CSV file with image file names and labels.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, 'images', str(idx) + '.tif')
        image = Image.open(img_name)

        label = int(self.data_df.iloc[idx, 1])  # Assuming label is in the second column of the CSV file

        if self.transform:
            image = self.transform(image)

        return image, label

def create_data_loaders(data_dir, train_csv, test_csv, valid_csv, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PolypDataset(data_dir=data_dir,
                                 csv_file=train_csv,
                                 transform=transform)

    test_dataset = PolypDataset(data_dir=data_dir,
                                csv_file=test_csv,
                                transform=transform)

    valid_dataset = PolypDataset(data_dir=data_dir,
                                 csv_file=valid_csv,
                                 transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, valid_loader

