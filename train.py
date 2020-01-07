import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_utils import LanePoseDataset
from utils.utils import ToCustomTensor, TransCropHorizon, weights_init
from losses import weighted_l1
from models import VanillaCNN, SpectralDropoutCNN

# System
path_to_home = os.environ['HOME']
path_to_proj = os.path.join(path_to_home, 'spectral-dropout')
path_dataset = os.path.join(path_to_proj, 'dataset_LanePose')
path_save = os.path.join(path_to_proj, 'saved_models')

# Model parameters
IMAGE_RESOLUTION = 64
GREYSCALE = True

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 1
GPU = 1

if torch.cuda.is_available():
    device = torch.device(GPU)
else:
    device = torch.device('cpu')

# Define required transformations of dataset
transforms = transforms.Compose([
    transforms.Resize(IMAGE_RESOLUTION),
    TransCropHorizon(0.5, set_black=False),
    transforms.Grayscale(num_output_channels=1),
    ToCustomTensor(),
    ])

# Load data & create dataset
log_names = sorted(next(os.walk(path_dataset))[1])


train_dict = {}
for idx, log_name in enumerate(log_names, 1):
    log_path = os.path.join(path_dataset, log_name)
    csv_path = os.path.join(log_path, 'output_pose.csv')
    img_path = os.path.join(log_path, 'images')
    train_dict['ts_' + str(idx)] = LanePoseDataset(csv_file=csv_path,
                                                   img_path=img_path,
                                                   transform=transforms)

training_set = torch.utils.data.ConcatDataset(train_dict.values())
training_loader = DataLoader(training_set,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS)

# Define the model
model = SpectralDropoutCNN(as_gray=GREYSCALE)
model.double().to(device=device)
model.apply(weights_init)

print(model)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=1e-3)

# Training
model_save_name = ''.join(model.name)  # later add configuration
epoch_steps = len(training_loader)
for epoch in range(NUM_EPOCHS):

    time_epoch_start = time.time()

    epoch_loss_list = []
    epoch_batch_list = []
    epoch_epoch_list = []


    for idx, (images, poses) in enumerate(training_loader):
        print(images.shape)
        # TODO either do this on data or remove it
        # Normalize pose theta to range [-1, 1]
        poses[:, 1] = poses[:, 1]/3.1415

        # Assign Tensors to Cuda device
        images = images.double().to(device=device)
        poses = poses.to(device=device)

        # Feedforward
        outputs = model(images)

        # Compute loss
        train_loss = weighted_l1(poses, outputs)

        epoch_loss_list.append(train_loss.item())
        epoch_epoch_list.append(epoch)

        # Backpropagate
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print progress
        if (idx + 1) % BATCH_SIZE == 0:
            print('Epoch [{}/{}], Step [{}/{}], Item [{}/{}], Loss: {:.6f}' #, Accuracy: {:.2f}%'
                  .format(epoch + 1, NUM_EPOCHS,
                          idx + 1, epoch_steps,
                          BATCH_SIZE*(idx+1), len(training_set),
                          epoch_loss_list[-1]))

    # Backup model after every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model, os.path.join(path_save, model_save_name))

# Save model
torch.save(model, os.path.join(path_save, model_save_name))
print('Model saved:', os.path.join(path_save, model_save_name))
