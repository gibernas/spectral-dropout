import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset



class LanePoseDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None):
        """
        This custom dataloader loads a batch of data and returns the data in
        tensor format
        Args:
            csv_file (string): Path to the csv file.
            img_path (string): Directory with all the images.
            transform: Optional transform to be applied on a batch.
        """
        self.data = pd.read_csv(csv_file, header=0, engine='python')
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.data.iloc[idx, 0]

        # Add .jpg to image name if not yet existing
        if '.jpg' not in image_name:
            image_name = image_name + '.jpg'

        img_name = os.path.join(self.img_path, image_name)
        image = io.imread(img_name)

        pose = self.data.iloc[idx, 2:4]
        pose = np.array(pose)
        pose = pose.astype('float')

        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, pose

