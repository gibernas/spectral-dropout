import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def plot_hidden_old(activations_batch):
    """Helper function to display the activations"""
    fig = plt.figure(figsize=(24, 16))
    batch_size = activations_batch.shape[0]
    channels = activations_batch.shape[1]
    columns = 4
    rows = int(np.ceil(batch_size*channels/columns))
    for i, batch in enumerate(activations_batch):
        for j, channel in enumerate(batch):
            image = channel.detach().numpy()
            fig.add_subplot(rows, columns, i + 1 + j)
            plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.show()


def plot_hidden(writer, activations_batch, i, layer_name='default'):
    """Helper function to display the activations

    We expect grayscale images. If multiple channels are present, they refer to filter channels, not image channles"""

    num_filter_channels = activations_batch.shape[1]
    for idx in range(num_filter_channels):
        try:
            single_channel = activations_batch[:, idx, :, :]
            big_image = make_grid(single_channel[:, None, :, :])
            name = ''.join([str(i), '_', str(idx), '_', layer_name])
            writer.add_image(name, big_image, 0)
        except (RuntimeError, TypeError):
            print('Error with shape: ')
            print(activations_batch.shape)
            print(big_image.shape)
        except IndexError:
            pass



