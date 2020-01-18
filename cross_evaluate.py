import os
import pandas as pd

import torch

from losses import weighted_l1
from utils.data_utils import get_test_data_loaders
from utils.utils import get_parser

# System
path_to_home = os.environ['HOME']
path_to_proj = os.path.join(path_to_home, 'spectral-dropout')
path_load = os.path.join(path_to_proj, 'saved_models')

# Model parameters
GREYSCALE = True

# Get a parser for the evaluation settings
parser = get_parser()


def evaluate(model, test_loader, dev=torch.device('cpu')):
    model.eval()
    test_loss = 0
    print('loader of size: %s' % len(test_loader))
    with torch.no_grad():
        print(test_loader.dataset.datasets)
        for batch_idx, (images, poses) in enumerate(test_loader):
            # Assign Tensors to Cuda device
            images, poses = images.double().to(device=dev), poses.to(device=dev)
            # Normalize pose theta to range [-1, 1]
            poses[:, 1] = poses[:, 1] / 3.1415
            # Feedforward
            outputs = model(images)
            test_loss += weighted_l1(poses, outputs).item()
    print('Test loss: %s' % (test_loss / len(test_loader)))
    return test_loss / len(test_loader)


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device(args.gpu if use_cuda else "cpu")
    print('Using device %s' % device)

    datasets_list = []
    if args.host == 'rudolf':
        path_dataset_root = '/media/9tb/ds'
    elif args.host == 'local':
        path_dataset_root = path_to_proj
    elif args.host == 'leonhard':
        path_dataset_root = '/cluster/scratch/gibernas'
    else:
        raise RuntimeError('The specified host is not supported')

    datasets_list.append(os.path.join(path_dataset_root, 'dataset_LanePose_real'))
    datasets_list.append(os.path.join(path_dataset_root, 'dataset_LanePose_sim'))

    # Get all models named *final*.pt
    models = []
    for subdir, dirs, files in os.walk(path_load):
        for filename in files:
            filepath = os.path.join(subdir, filename)
            if filepath.endswith('.pt') and 'final' in filepath:
                models.append((filepath, os.path.splitext(filename)[0]))

    # Create data loaders for: only sim / only real / both
    test_loaders = get_test_data_loaders(datasets_list, args)

    # Test all models on all datasets
    test_results = []
    print('#' * 20 + '\nStarting evaluations..')
    for model_entry in models:
        print('#'*20 + "\nEvaluating model %s" % model_entry[1])
        model = torch.load(model_entry[0], map_location=device)
        model_results = [model_entry[1]]
        for test_loader in test_loaders:
            print('dataset: %s' % test_loader)
            model_dataset_loss = evaluate(model, test_loader, dev=device)
            model_results.append(model_dataset_loss)
        test_results.append(model_results)
    print('#' * 20 + '\n' + 'Evaluation ended!\n' + '#' * 20)

    # Save scores
    df = pd.DataFrame(test_results, columns=['ModelName', 'RealDataset', 'SimDataset'])
    path_csv = os.path.join(path_to_proj, 'results', 'evaluation_results.csv')
    df.to_csv(path_or_buf=path_csv, index=False)
