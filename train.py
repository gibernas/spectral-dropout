import csv
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from losses import weighted_l1
from models import VanillaCNN, SpectralDropoutCNN, SpectralDropoutEasyCNN
from utils.data_utils import get_data_loaders
from utils.utils import get_parser

# System
path_to_home = os.environ['HOME']
path_to_proj = os.path.join(path_to_home, 'spectral-dropout')
# path_dataset_real = os.path.join(path_to_proj, 'dataset_LanePose')
path_dataset_real = '/media/9tb/ds/dataset_LanePose'
# path_dataset_sim = os.path.join(path_to_proj, 'dataset_LanePose_sim')
path_dataset_sim = '/media/9tb/ds/dataset_LanePose_sim'
path_save = os.path.join(path_to_proj, 'saved_models')

# Model parameters
GREYSCALE = True
WEIGHT_DECAY = 1e-3

# Get a parser for the training settings
parser = get_parser()


def train(model, opt, train_loader, dev=torch.device('cpu')):
    model.train()
    total_training_loss = 0

    for batch_idx, (images, poses) in enumerate(train_loader):
        # Assign Tensors to Cuda device
        images, poses = images.double().to(device=dev), poses.to(device=dev)

        # Normalize pose theta to range [-1, 1]
        poses[:, 1] = poses[:, 1]/3.1415

        # Feedforward
        outputs = model(images)

        # Compute loss
        train_loss = weighted_l1(poses, outputs)
        total_training_loss += train_loss.item()

        epoch_loss_list.append(train_loss.item())
        epoch_epoch_list.append(epoch)

        # Backpropagate
        opt.zero_grad()
        train_loss.backward()
        opt.step()

    return total_training_loss / len(train_loader)


def test(model, val_loader, dev=torch.device('cpu')):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (images, poses) in enumerate(val_loader):
            # Assign Tensors to Cuda device
            images, poses = images.double().to(device=dev), poses.to(device=dev)

            # Normalize pose theta to range [-1, 1]
            poses[:, 1] = poses[:, 1] / 3.1415

            # Feedforward
            outputs = model(images)
            val_loss += weighted_l1(poses, outputs).item()

    return val_loss / len(val_loader)


if __name__ == "__main__":

    args = parser.parse_args()

    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device(args.gpu if use_cuda else "cpu")
    print('Using device %s' % device)

    # Define the model
    if args.model == "VanillaCNN":
        model = VanillaCNN(as_gray=GREYSCALE)
        model.double().to(device=device)
    elif args.model == "SpectralDropoutCNN":
        model = SpectralDropoutCNN(as_gray=GREYSCALE)
        model.double().to(device=device)
    elif args.model == "SpectralDropoutEasyCNN":
        model = SpectralDropoutEasyCNN(as_gray=GREYSCALE, dev=device)
        model.double().to(device=device)
    else:
        raise RuntimeError ('You did not provide a valid model to train!')
    time_start = time.time()
    model_save_name = ''.join([model.name, str(time_start), '_lr', str(args.lr),
                               '_bs', str(args.batch_size), '_totepo', str(args.epochs)])
    print(model)

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                weight_decay=1e-3)

    list_path_datasets = []
    if 'real' in args.dataset:
        list_path_datasets.append(path_dataset_real)
    if 'sim' in args.dataset:
        list_path_datasets.append(path_dataset_sim)
    training_loader, validation_loader = get_data_loaders(list_path_datasets, args)

    writer = SummaryWriter()
    model_data_log = []
    for epoch in range(args.epochs):
        time_epoch_start = time.time()
        print("Starting epoch %s" % epoch)

        epoch_loss_list = []
        epoch_batch_list = []
        epoch_epoch_list = []

        training_loss = train(model, optimizer, training_loader, dev=device)
        writer.add_scalar('Loss/train', training_loss, epoch)

        validation_loss = test(model, validation_loader, dev=device)
        writer.add_scalar('Loss/val', validation_loss, epoch)

        # Compute some training stats and log them
        time_epoch_end = time.time()
        time_epoch = time_epoch_end - time_epoch_start
        writer.add_scalar('Epoch time', time_epoch, epoch)
        model_data_log.append([epoch, training_loss, validation_loss, time_epoch])

        # Backup model after every 10 epochs
        if (epoch + 1) % 10 == 0:
            current_model_save_name = ''.join([model_save_name, '_epo%s' % epoch, '.pt'])
            torch.save(model, os.path.join(path_save, current_model_save_name))
            print('Model saved on epoch %s' % epoch)

    writer.close()

    # Write config csv
    config_name = '_'.join([model_save_name, str(args.dataset), 'config.csv'])
    config_path = os.path.join(path_save, config_name)
    with open(config_path, 'w') as csv_config:
        wr = csv.writer(csv_config, quoting=csv.QUOTE_ALL)
        wr.writerow(['dataset', 'learning_rate', 'batch_size', 'num_epochs'])
        wr.writerow([args.dataset, args.lr, args.batch_size, args.epochs])

    # Write training data csv
    data_name = ''.join([model_save_name, '_config.csv'])
    data_path = os.path.join(path_save, data_name)
    with open(data_path, 'w') as csv_data:
        wr = csv.writer(csv_data, quoting=csv.QUOTE_ALL)
        wr.writerow(['epoch', 'training_loss', 'validation_loss', 'duration [s]'])
        for row in model_data_log:
            wr.writerow(row)
