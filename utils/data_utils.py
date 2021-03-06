import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from utils.utils import ToCustomTensor, TransCropHorizon


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
        # Remove NaNs, if the dataset was bad (i.e. out of lane)
        self.data = self.data.dropna(subset=['centerDistance', 'relativeHeading'])
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


def get_data_loaders(list_path_datasets, args):

    # Define required transformations of dataset
    tfs = transforms.Compose([
        transforms.Resize(args.image_res),
        TransCropHorizon(0.5, set_black=False),
        transforms.Grayscale(num_output_channels=1),
        ToCustomTensor(),
    ])

    dataset_dict = {}
    for path_dataset in list_path_datasets:
        print('Looking in: %s' % path_dataset)
        env = path_dataset.split('_')[-1]
        # Load data & create dataset
        log_names = sorted(next(os.walk(path_dataset))[1])

        for idx, log_name in enumerate(log_names, 1):
            if 'test' in log_name:
                pass
            else:
                log_path = os.path.join(path_dataset, log_name)
                csv_path = os.path.join(log_path, 'output_pose.csv')
                img_path = os.path.join(log_path, 'images')
                dataset_dict['ts_' + str(idx) + '_' + env] = LanePoseDataset(csv_file=csv_path,
                                                                             img_path=img_path,
                                                                             transform=tfs)

    dataset = torch.utils.data.ConcatDataset(dataset_dict.values())

    validation_split = args.validation_split

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))

    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(123)  # So that each model has the same training/validation datasets
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.workers,
                                 sampler=train_sampler)

    validation_loader = DataLoader(dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.workers,
                                   sampler=valid_sampler)
    return training_loader, validation_loader


def get_test_data_loaders(list_path_datasets, args):
    # Define required transformations of dataset
    tfs = transforms.Compose([
        transforms.Resize(args.image_res),
        TransCropHorizon(0.5, set_black=False),
        transforms.Grayscale(num_output_channels=1),
        ToCustomTensor(),
    ])
    dataset_dict = {}
    for path_dataset in list_path_datasets:
        single_dataset_dict = {}
        print('Looking in: %s' % path_dataset)
        env = path_dataset.split('_')[-1]
        # Load data & create dataset
        log_names = sorted(next(os.walk(path_dataset))[1])

        for idx, log_name in enumerate(log_names, 1):
            if 'test' not in log_name:
                pass
            else:
                log_path = os.path.join(path_dataset, log_name)
                csv_path = os.path.join(log_path, 'output_pose.csv')
                img_path = os.path.join(log_path, 'images')
                single_dataset_dict['ts_' + str(idx) + '_' + env] = LanePoseDataset(csv_file=csv_path,
                                                                                    img_path=img_path,
                                                                                    transform=tfs)

        dataset_dict[env] = torch.utils.data.ConcatDataset(single_dataset_dict.values())

    real_test_loader = DataLoader(dataset_dict['real'],
                                  batch_size=args.batch_size,
                                  num_workers=args.workers,
                                  )

    sim_test_loader = DataLoader(dataset_dict['sim'],
                                 batch_size=args.batch_size,
                                 num_workers=args.workers,
                                 )

    return [real_test_loader, sim_test_loader]

